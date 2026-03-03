import re
import json


def tag_count_reward(predict_str: str, with_answer: bool = True) -> float:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags_with_answer(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.2
        if text.count("\n</think>\n") == 1:
            count += 0.2
        if text.count("\n<answer>\n") == 1:
            count += 0.2
        if text.count("\n</answer>") == 1:
            count += 0.2
        
        if count == 0.8:
            answer_rest = text.split("\n</answer>")[-1]
            if len(answer_rest) < 5:
                count += 0.2
        return count

    def count_tags_without_answer(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        
        if count == 0.5:
            answer_full = text.split("\n</think>\n")[-1]
            boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            match = re.search(boxed_pattern, answer_full)
            if match:
                count += 0.25
                answer_rest = answer_full[match.end():]
                if len(answer_rest) < 5:
                    count += 0.25
        return count

    if with_answer:
        return count_tags_with_answer(predict_str)
    else:
        return count_tags_without_answer(predict_str)


def extract_answer_content(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""


def json_match(output_str):
    try:
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, output_str, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            result = data["result"][0]
            score = int(result["score"])
            analysis = result["analysis"]
        else:
            score = None
            analysis = None
    except Exception as e:
        print(e)
        score = None
        analysis = None
    return score, analysis


def re_pattern_match(output_str):
    score_pattern = r'"score"\s*:\s*(\d+)'
    analysis_pattern = r'"analysis"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'

    score_match = re.search(score_pattern, output_str)
    analysis_match = re.search(analysis_pattern, output_str, re.DOTALL)

    score = score_match.group(1) if score_match else None
    analysis = analysis_match.group(1) if analysis_match else None
    return score, analysis

