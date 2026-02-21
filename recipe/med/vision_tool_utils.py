import base64
import io
import json
import logging
import threading
from typing import Any

import requests
from PIL import Image

DEFAULT_TIMEOUT = 10

logger = logging.getLogger(__name__)


def encode_base64_image(image: Image.Image, format: str = "JPEG") -> str:
    """
    Encodes a PIL.Image object to a base64 string.

    Args:
        image: The PIL.Image object to encode.
        format: The image format to use for encoding (e.g., "PNG", "JPEG").
                Defaults to "JPEG".

    Returns:
        A base64 encoded string of the image.
    """
    if format.upper() == "JPEG":
        if image.mode in ("RGBA", "P"):
            # Convert RGBA or P to RGB before saving as JPEG
            # RGBA will discard the alpha channel.
            # P will convert the palette-indexed colors to direct RGB values.
            image = image.convert("RGB")
        elif image.mode == "L":  # Grayscale image
            # Although L mode can be saved as JPEG, converting to RGB can sometimes
            # avoid issues if the JPEG encoder has a preference for color images,
            # or if downstream expects color images. For simplicity, we convert to RGB.
            # If strict grayscale output for JPEG is desired, this line could be removed.
            image = image.convert("RGB")

    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_base64_to_image(base64_string: str) -> Image.Image:
    """
    Decodes a base64 string to a PIL.Image object.

    Args:
        base64_string: The base64 encoded string of the image.

    Returns:
        A PIL.Image object.
    """
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)

    # Create a BytesIO object from the image data bytes
    buffered = io.BytesIO(image_data)

    # Open the image from the BytesIO object
    image = Image.open(buffered)

    return image


def call_vision_tool_api(
    vision_tool_service_url: str,
    image_b64: str,
    parameters: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str | None, dict[str, Any]]:
    try:
        image_bytes = base64.b64decode(image_b64)

        # 2. Prepare the multipart/form-data payload
        # 'files' is used for the image file itself
        files = {"image_file": ("image.png", image_bytes, "image/png")}
        processed_data = {}
        for k, v in parameters.items():
            if v is None:
                continue

            if isinstance(v, (dict, list)):
                processed_data[k] = json.dumps(v, ensure_ascii=False)
            else:
                processed_data[k] = str(v)

        # 3. Make the POST request
        response = requests.post(
            vision_tool_service_url,
            files=files,
            data=processed_data,
            timeout=(5, timeout),  # (connect_timeout, read_timeout)
        )

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # 4. Process the successful response
        response_data = response.json()

        processed_image_b64 = response_data.pop("processed_image_b64", None)

        return processed_image_b64, response_data

    except requests.exceptions.Timeout:
        return None, {"message": "Request timeout"}
    except requests.exceptions.ConnectionError:
        return None, {
            "message": "Connection failed, stop calling the tool and answer the question directly."
        }
    except requests.exceptions.HTTPError as e:
        return None, {"message": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        return None, {"message": f"Unexpected error: {str(e)}"}


def perform_single_vision_tool(
    vision_tool_service_url: str,
    image_b64: str,
    parameters: dict[str, Any],
    concurrent_semaphore: threading.Semaphore | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[str | None, dict[str, Any]]:
    if concurrent_semaphore:
        with concurrent_semaphore:
            processed_image_b64, metadata = call_vision_tool_api(
                vision_tool_service_url=vision_tool_service_url,
                image_b64=image_b64,
                parameters=parameters,
                timeout=timeout,
            )
    else:
        processed_image_b64, metadata = call_vision_tool_api(
            vision_tool_service_url=vision_tool_service_url,
            image_b64=image_b64,
            parameters=parameters,
            timeout=timeout,
        )

    return processed_image_b64, metadata
