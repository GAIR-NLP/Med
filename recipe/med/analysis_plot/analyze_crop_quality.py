#!/usr/bin/env python3
"""
Analyze whether cropped images are beneficial for solving questions.

This script:
1. Loads jsonl files with matched samples
2. Extracts bbox_2d from the last <tool_call> in messages
3. Crops images using image_crop_and_zoom_in_tool
4. Asks Gemini whether the cropped image is beneficial for the task
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import vision_tool_fn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Gemini API setup
from google import genai
from google.genai import types
from vision_tool_fn import image_crop_and_zoom_in_tool

MODEL_NAME = "gemini-3-pro-preview"
BASE_URL = "http://thirdpart-proxy-prod.xaminim.com/v1/proxy/gemini"
API_TOKEN = "zBh8ZDlVCo4ftiwBMsWKpBeqCqVUgHN9Iov9I1m6U24NC9zYSeGbTquVOvm1ghXrHdBgrR9wO4nxhsU42taB9HYdeDl8QHWd3qohT1U-mpHXBA7HCs-yugO0RmpLokp6cUsGfrJvI_CxF88WXslp_2aaarxsYlq6PjOkQeXcPis="
billing_token = "d26257bafad37536eefd11e402b64e75ea85a7d52613af1cf6d1d675c1f2d2ec"
API_KEY = f"{API_TOKEN}:{billing_token}"

# Prompt template for Gemini
ANALYSIS_PROMPT = """You are an expert in visual question answering. I will show you:
1. The original image
2. A cropped region from that image
3. The question that needs to be answered
4. The ground truth answer
5. The model's response process

Your task is to evaluate whether the cropped region in the tool call was beneficial for the model to answer the question during its response process.

Question: {question}

Ground Truth Answer: {answer}

Model's Response Process:
{response}

Please analyze:

1. Does the cropped region contain relevant information for answering the question?
2. Did the crop help the model in its reasoning process to arrive at the answer?
3. Was the crop well-positioned and properly sized for this task?

Respond with a JSON object with the following format:
{{
    "is_beneficial": true/false,
    "relevance_score": <1-5, where 5 is highly relevant>,
    "crop_quality_score": <1-5, where 5 is perfectly cropped>,
    "reasoning": "<your detailed explanation>"
}}

Only output the JSON, no other text.
"""


def resize_image_short_side(img: Image.Image, max_short_side: int = 512) -> Image.Image:
    """Resize image so that the shorter side is at most max_short_side pixels.

    Args:
        img: PIL Image
        max_short_side: Maximum length for the shorter side (default: 512)

    Returns:
        Resized PIL Image
    """
    width, height = img.size

    # Find the shorter side
    short_side = min(width, height)

    # If already smaller than max, return as-is
    if short_side <= max_short_side:
        return img

    # Calculate scaling factor
    scale = max_short_side / short_side

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def load_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
    """Load jsonl file."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def extract_tool_call_bbox(messages: list[dict[str, Any]]) -> tuple[list[float], str] | None:
    """Extract bbox_2d from the last <tool_call> in messages.

    Args:
        messages: List of message dictionaries

    Returns:
        Tuple of (bbox_2d, label) or None if not found
    """
    # Search backwards through messages
    for message in reversed(messages):
        content = message.get("content", "")

        # Extract text from content (handle both string and list formats)
        text = extract_text_from_content(content)
        if not text:
            continue

        # Find <tool_call> tags
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(tool_call_pattern, text, re.DOTALL)

        if not matches:
            continue

        # Take the last tool_call in this message
        tool_call_str = matches[-1].strip()

        try:
            # Parse the JSON
            tool_call_data = json.loads(tool_call_str)

            # Check if it's image_crop_and_zoom_in_tool
            if tool_call_data.get("name") == "image_crop_and_zoom_in_tool":
                arguments = tool_call_data.get("arguments", {})
                bbox_2d = arguments.get("bbox_2d")
                label = arguments.get("label", "cropped_region")

                if bbox_2d and len(bbox_2d) == 4:
                    return bbox_2d, label
        except json.JSONDecodeError:
            continue

    return None


def extract_text_from_content(content: Any) -> str:
    """Extract text from content field which can be string or list of dicts.

    Args:
        content: Either a string or list of dicts with 'type' and 'text' fields

    Returns:
        Extracted text string
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from list of dicts
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return " ".join(text_parts)
    else:
        return ""


def extract_response_from_messages(messages: list[dict[str, Any]]) -> str:
    """Extract concatenated response from messages starting from index 1.

    Args:
        messages: List of message dictionaries

    Returns:
        Concatenated text from all messages starting from index 1
    """
    if len(messages) <= 1:
        return ""

    response_parts = []
    # Start from index 1 (skip first message)
    for message in messages[1:]:
        content = message.get("content", "")
        text = extract_text_from_content(content)
        if text:
            response_parts.append(text)

    return "\n\n".join(response_parts)


def ask_gemini_about_crop(
    original_image: Image.Image,
    cropped_image: Image.Image,
    question: str,
    answer: str,
    response: str,
) -> dict[str, Any]:
    """Ask Gemini whether the cropped image is beneficial.

    Args:
        original_image: Original PIL Image
        cropped_image: Cropped PIL Image
        question: The question to answer
        answer: The ground truth answer
        response: The model's response process

    Returns:
        Dictionary with Gemini's analysis
    """
    client = genai.Client(api_key=API_KEY, http_options={"base_url": BASE_URL})

    try:
        prompt = ANALYSIS_PROMPT.format(question=question, answer=answer, response=response)

        # Prepare multimodal content
        # Note: The google.genai library might need images in a specific format
        # We'll use PIL images directly
        contents = [
            "Original Image:",
            original_image,
            "\nCropped Region:",
            cropped_image,
            "\n" + prompt,
        ]

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )

        response_text = response.text.strip()

        # Try to parse JSON from response
        # Sometimes the model might wrap JSON in markdown code blocks
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {"error": "Failed to parse JSON from response", "raw_response": response_text}

    except Exception as e:
        return {"error": str(e)}


def process_single_sample(sample: dict[str, Any], base_dir: str) -> dict[str, Any]:
    """Process a single sample: extract bbox, crop image, ask Gemini.

    Args:
        sample: Sample data from jsonl
        base_dir: Base directory containing jsonl file (for resolving image paths)

    Returns:
        Sample with added analysis results
    """
    try:
        # Load original image
        image_path = sample.get("image_path")
        if not image_path:
            return {**sample, "analysis": {"error": "No image_path found"}}

        full_image_path = os.path.join(base_dir, image_path)
        if not os.path.exists(full_image_path):
            return {**sample, "analysis": {"error": f"Image not found: {full_image_path}"}}

        original_image = Image.open(full_image_path)
        # Resize so that short side is at most 512
        # original_image = resize_image_short_side(original_image, max_short_side=512)

        # Extract bbox from messages
        messages = sample.get("messages", [])
        bbox_result = extract_tool_call_bbox(messages)

        if not bbox_result:
            return {**sample, "analysis": {"error": "No tool_call with bbox_2d found"}}

        bbox_2d, label = bbox_result

        # Crop image using vision_tool_fn
        crop_result = image_crop_and_zoom_in_tool(
            img=original_image, bbox_2d=bbox_2d, label=label, reward_strategy="baseline"
        )

        if crop_result["status"] == "error" or crop_result["image"] is None:
            return {
                **sample,
                "analysis": {
                    "error": f"Crop failed: {crop_result.get('message', 'Unknown error')}",
                    "bbox_2d": bbox_2d,
                },
            }

        cropped_image = crop_result["image"]
        # Resize cropped image as well
        # cropped_image = resize_image_short_side(cropped_image, max_short_side=512)

        # Extract question and answer from original_data
        try:
            question = sample["original_data"]["extra_info"]["question"]
            answer = sample["original_data"]["extra_info"]["answer"]
        except (KeyError, TypeError):
            return {
                **sample,
                "analysis": {
                    "error": "Could not extract question or answer from original_data",
                    "bbox_2d": bbox_2d,
                },
            }

        if not question or not answer:
            return {
                **sample,
                "analysis": {"error": "Question or answer is empty", "bbox_2d": bbox_2d},
            }

        # Extract response from messages (starting from index 1)
        response = extract_response_from_messages(messages)

        if not response:
            return {
                **sample,
                "analysis": {
                    "error": "Could not extract response from messages",
                    "bbox_2d": bbox_2d,
                },
            }

        # Ask Gemini
        gemini_result = ask_gemini_about_crop(
            original_image=original_image,
            cropped_image=cropped_image,
            question=question,
            answer=answer,
            response=response,
        )

        # Add metadata to result
        analysis = {
            "bbox_2d": bbox_2d,
            "label": label,
            "crop_status": crop_result["status"],
            "crop_message": crop_result["message"],
            "question": question,
            "answer": answer,
            "response": response,
            **gemini_result,
        }

        return {**sample, "analysis": analysis}

    except Exception as e:
        return {**sample, "analysis": {"error": f"Exception: {str(e)}"}}


def analyze_jsonl_file(
    jsonl_path: str, output_path: str = None, max_workers: int = 3, max_samples: int = None
) -> None:
    """Analyze all samples in a jsonl file.

    Args:
        jsonl_path: Path to input jsonl file
        output_path: Path to output jsonl file (default: input_analyzed.jsonl)
        max_workers: Number of concurrent workers for Gemini API
        max_samples: Maximum number of samples to process (for testing)
    """
    if output_path is None:
        base_name = os.path.splitext(jsonl_path)[0]
        output_path = f"{base_name}_analyzed.jsonl"

    print(f"Loading samples from: {jsonl_path}")
    samples = load_jsonl(jsonl_path)

    if max_samples:
        samples = samples[:max_samples]
        print(f"Processing first {max_samples} samples (test mode)")

    print(f"Total samples to process: {len(samples)}")

    base_dir = os.path.dirname(jsonl_path)

    results = []

    # Process with ThreadPoolExecutor for concurrent Gemini API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_sample, sample, base_dir): sample for sample in samples
        }

        for future in tqdm(as_completed(futures), total=len(samples), desc="Analyzing samples"):
            result = future.result()
            results.append(result)

    # Sort results by idx to maintain original order
    results.sort(key=lambda x: x.get("idx", 0))

    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print summary
    successful = sum(1 for r in results if "error" not in r.get("analysis", {}))
    beneficial = sum(1 for r in results if r.get("analysis", {}).get("is_beneficial", False))

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Successfully analyzed: {successful}")
    print(
        f"Beneficial crops: {beneficial} ({beneficial/successful*100:.1f}%)"
        if successful > 0
        else "Beneficial crops: 0"
    )
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze whether cropped images are beneficial for solving questions"
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output jsonl file (default: <input>_analyzed.jsonl)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent workers for Gemini API (default: 3)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    args = parser.parse_args()

    analyze_jsonl_file(args.jsonl, args.output, args.workers, args.max_samples)


# Example usage:
# python3 analyze_crop_quality.py \
#     --jsonl matched_samples_jsonl/qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew/qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew.jsonl \
#     --workers 3 \
#     --max_samples 10
# python3 recipe/o3/plot_v3/analyze_crop_quality.py       --jsonl matched_samples_parquet/qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew/qwen25vl_instruct_75_50_natural_0.75_toolcall_0.5_no_rew.jsonl       --workers 5
# python3 recipe/o3/plot_v3/analyze_crop_quality.py       --jsonl matched_samples_parquet/qwen3vl_instruct_75_50_qwen3vl_natural_0.75_toolcall_0.5_no_rew/qwen3vl_instruct_75_50_qwen3vl_natural_0.75_toolcall_0.5_no_rew.jsonl       --workers 5
if __name__ == "__main__":
    main()
