import logging
import math
from math import ceil, floor

from PIL import Image
from qwen_vl_utils import fetch_image

logger = logging.getLogger("vision_tool")


MAX_RATIO = 200
TOOL_PENALTY = 0.0
TOOL_REWARD = 1.0

# Constants for the new image_crop_and_zoom_in_tool
FACTOR = 28
MIN_BBOX_SIZE = 28
MAX_TOOL_RESPONSE_TOKENS = 1024
DEFAULT_MIN_PIXELS = 256 * 28 * 28
MIN_PIXELS_THRESHOLD = 4 * 28 * 28
MAX_PIXELS_THRESHOLD = MAX_TOOL_RESPONSE_TOKENS * 28 * 28


def _baseline_reward_compute() -> float:
    """
    Baseline strategy: returns a fixed positive reward if the tool executes successfully
    (i.e., does not exit early due to an error).
    """
    return TOOL_REWARD


def smart_crop_and_zoom(
    img: Image.Image,
    crop_box: tuple[float, float, float, float],
    zoom_factor: float | None = None,
    output_resolution_limit: int | None = 256,
    reward_strategy: str = "baseline",
) -> dict:
    tool_reward = TOOL_PENALTY
    notifications = []  # A list to accumulate detailed warning messages

    try:
        width, height = img.size
        x1, y1, x2, y2 = crop_box

        # --- Coordinate system and order correction logic ---
        # Check coordinate formats (order matters: check old format first to avoid conflicts)
        is_old_relative = all(isinstance(c, (int, float)) and 0.0 <= c <= 1.0 for c in crop_box)
        is_new_relative = (
            all(isinstance(c, (int, float)) and 0 <= c <= 1000 for c in crop_box)
            and not is_old_relative
        )

        if is_new_relative:
            # Convert [0, 1000] integers to relative coordinates [0, 1]
            rel_x1, rel_y1, rel_x2, rel_y2 = (x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0)
            abs_x1, abs_y1, abs_x2, abs_y2 = (
                rel_x1 * width,
                rel_y1 * height,
                rel_x2 * width,
                rel_y2 * height,
            )
        elif is_old_relative:
            abs_x1, abs_y1, abs_x2, abs_y2 = (
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height,
            )
        else:
            notifications.append(f"Input {crop_box} was interpreted as absolute pixel coordinates.")
            abs_x1, abs_y1, abs_x2, abs_y2 = (x1, y1, x2, y2)

        original_x_coords = (int(abs_x1), int(abs_x2))
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
            notifications.append(
                # f"Input x-coordinates were inverted and swapped from {original_x_coords} to {(int(abs_x1), int(abs_x2))}."
                f"Input x-coordinates were inverted and swapped from {(x1, x2)} to {(x2, x1)}."
            )

        original_y_coords = (int(abs_y1), int(abs_y2))
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
            notifications.append(
                # f"Input y-coordinates were inverted and swapped from {original_y_coords} to {(int(abs_y1), int(abs_y2))}."
                f"Input y-coordinates were inverted and swapped from {(y1, y2)} to {(y2, y1)}."
            )

        int_coords = tuple(map(int, (abs_x1, abs_y1, abs_x2, abs_y2)))
        final_x1, final_y1 = max(0, int_coords[0]), max(0, int_coords[1])
        final_x2, final_y2 = min(width, int_coords[2]), min(height, int_coords[3])
        final_coords = (final_x1, final_y1, final_x2, final_y2)

        if final_coords != int_coords:
            notifications.append(
                # f"Coordinates {int_coords} were clamped to {final_coords} to fit within image dimensions {width}x{height}."
                f"Coordinates {crop_box} were clamped to fit within image dimensions."
            )

        if final_x1 >= final_x2 or final_y1 >= final_y2:
            return {
                "image": img,
                "status": "error",
                "message": f"Invalid crop dimensions. The calculated crop area {crop_box} has zero or negative size.",
                "tool_reward": tool_reward,
            }

        cropped_img = img.crop(final_coords)

        # --- Internal & Silent Logic: Reward, Fetching, Zoom ---
        if reward_strategy == "baseline":
            tool_reward = _baseline_reward_compute()
        else:
            raise ValueError(f"Unsupported reward strategy: {reward_strategy}")
        cropped_width, cropped_height = cropped_img.size
        if cropped_width < FACTOR or cropped_height < FACTOR:
            cropped_img = fetch_image({"image": cropped_img})
        elif max(cropped_width, cropped_height) / min(cropped_width, cropped_height) > MAX_RATIO:
            cropped_img = fetch_image({"image": cropped_img})

        if zoom_factor is None:
            zoom_factor = min(width / cropped_width, height / cropped_height)
        elif zoom_factor <= 0:
            zoom_factor = 1.0

        # --- Build return message ---
        def build_return_message(base_msg, final_image):
            final_status = "warning" if notifications else "success"
            if notifications:
                message = base_msg + "\n\nNotifications:\n\n" + "\n\n".join(notifications)
            else:
                message = base_msg
            return {
                "image": final_image,
                "status": final_status,
                "message": message,
                "tool_reward": tool_reward,
            }

        if zoom_factor == 1.0:
            return build_return_message("Image cropped successfully with no zoom.", cropped_img)

        ideal_width, ideal_height = (
            int(cropped_img.size[0] * zoom_factor),
            int(cropped_img.size[1] * zoom_factor),
        )
        final_width, final_height = ideal_width, ideal_height
        if output_resolution_limit and max(ideal_width, ideal_height) > output_resolution_limit:
            downscale_ratio = output_resolution_limit / max(ideal_width, ideal_height)
            final_width, final_height = (
                int(ideal_width * downscale_ratio),
                int(ideal_height * downscale_ratio),
            )

        if final_width < 1 or final_height < 1:
            return build_return_message(
                "Zoom resulted in an invalid image size (< 1px). Returning the cropped image without zoom.",
                cropped_img,
            )

        final_image = cropped_img.resize((final_width, final_height), Image.Resampling.LANCZOS)
        return build_return_message("Image cropped and zoomed successfully.", final_image)

    except Exception as e:
        return {
            "image": img,
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}, return the original image.",
            "tool_reward": tool_reward,
        }


def image_crop_and_zoom_in_tool(
    img: Image.Image,
    bbox_2d: tuple[float, float, float, float],
    label: str = "cropped_region",
    reward_strategy: str = "baseline",
) -> dict:
    """
    New implementation of crop and zoom with enhanced bbox validation and smart resizing.

    Args:
        img: PIL Image to process
        bbox_2d: Bounding box coordinates [x1, y1, x2, y2]
        label: Descriptive label for the cropped region
        reward_strategy: Reward computation strategy

    Returns:
        Dictionary with processed image, status, message, and tool_reward
    """
    tool_reward = TOOL_PENALTY

    try:
        img_width, img_height = img.size
        x1, y1, x2, y2 = bbox_2d

        # Determine if coordinates are relative (0-1000) or absolute (pixel values)
        max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))

        if max_coord <= 1000:
            # Coordinates are in relative format (0-1000), convert to absolute
            # First clamp to valid relative range
            x1 = max(0, min(1000, x1))
            y1 = max(0, min(1000, y1))
            x2 = max(0, min(1000, x2))
            y2 = max(0, min(1000, y2))

            abs_x1, abs_y1, abs_x2, abs_y2 = (
                x1 / 1000.0 * img_width,
                y1 / 1000.0 * img_height,
                x2 / 1000.0 * img_width,
                y2 / 1000.0 * img_height,
            )
        else:
            # Coordinates appear to be in absolute pixel format
            if (
                0 <= x1 <= img_width
                and 0 <= x2 <= img_width
                and 0 <= y1 <= img_height
                and 0 <= y2 <= img_height
            ):
                # All coordinates are within image bounds, use as-is
                abs_x1, abs_y1, abs_x2, abs_y2 = x1, y1, x2, y2
            else:
                # Coordinates exceed image bounds, convert to relative format first
                # Assume they were intended as relative but scaled incorrectly
                scale_factor = 1000.0 / max_coord
                rel_x1 = x1 * scale_factor
                rel_y1 = y1 * scale_factor
                rel_x2 = x2 * scale_factor
                rel_y2 = y2 * scale_factor

                abs_x1, abs_y1, abs_x2, abs_y2 = (
                    rel_x1 / 1000.0 * img_width,
                    rel_y1 / 1000.0 * img_height,
                    rel_x2 / 1000.0 * img_width,
                    rel_y2 / 1000.0 * img_height,
                )

        validated_bbox = _ensure_valid_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
        left, top, right, bottom = validated_bbox

        cropped_image = img.crop((left, top, right, bottom))
        new_w, new_h = _smart_resize(
            (right - left), (bottom - top), factor=FACTOR, min_pixels=DEFAULT_MIN_PIXELS
        )
        cropped_image = cropped_image.resize(
            (int(new_w), int(new_h)), resample=Image.Resampling.BICUBIC
        )

        # Validation logic similar to smart_crop_and_zoom
        cropped_width, cropped_height = cropped_image.size
        if cropped_width < FACTOR or cropped_height < FACTOR:
            cropped_image = fetch_image({"image": cropped_image})
        elif max(cropped_width, cropped_height) / min(cropped_width, cropped_height) > MAX_RATIO:
            cropped_image = fetch_image({"image": cropped_image})

        # Set reward based on strategy
        if reward_strategy == "baseline":
            tool_reward = _baseline_reward_compute()
        else:
            raise ValueError(f"Unsupported reward strategy: {reward_strategy}")

        return {
            "image": cropped_image,
            "status": "success",
            "message": f"Successfully cropped and processed region labeled as: {label}",
            "tool_reward": tool_reward,
        }

    except Exception as e:
        return {
            "image": None,
            "status": "error",
            "message": f"Error: {str(e)}",
            "tool_reward": tool_reward,
        }


def _smart_resize(
    height: int | float,
    width: int | float,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS_THRESHOLD,
    max_pixels: int = MAX_PIXELS_THRESHOLD,
) -> tuple[int | float, int | float]:
    """Smart resize image dimensions based on factor and pixel constraints"""
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _ensure_valid_bbox(
    left: float, top: float, right: float, bottom: float, img_width: int, img_height: int
) -> list[float]:
    """Ensure bbox is valid and meets minimum size requirements"""

    # Fix coordinate order issues - ensure left < right and top < bottom
    if left > right:
        left, right = right, left
    if top > bottom:
        top, bottom = bottom, top

    # Clamp coordinates to image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(img_width, right)
    bottom = min(img_height, bottom)

    # Double check after clamping - in case coordinates were way off
    if left >= right:
        # If still invalid, create a small valid box from the center
        center_x = left if left < img_width else img_width / 2
        left = max(0, center_x - MIN_BBOX_SIZE / 2)
        right = min(img_width, center_x + MIN_BBOX_SIZE / 2)

    if top >= bottom:
        # If still invalid, create a small valid box from the center
        center_y = top if top < img_height else img_height / 2
        top = max(0, center_y - MIN_BBOX_SIZE / 2)
        bottom = min(img_height, center_y + MIN_BBOX_SIZE / 2)

    height, width = bottom - top, right - left
    if height < MIN_BBOX_SIZE or width < MIN_BBOX_SIZE:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = MIN_BBOX_SIZE / min(height, width)
        new_half_height = ceil(height * ratio * 0.5)
        new_half_width = ceil(width * ratio * 0.5)
        new_left = floor(center_x - new_half_width)
        new_right = ceil(center_x + new_half_width)
        new_top = floor(center_y - new_half_height)
        new_bottom = ceil(center_y + new_half_height)

        # Ensure the resized bbox is within image bounds
        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(img_width, new_right)
        new_bottom = min(img_height, new_bottom)

        new_height = new_bottom - new_top
        new_width = new_right - new_left

        if new_height > MIN_BBOX_SIZE and new_width > MIN_BBOX_SIZE:
            return [new_left, new_top, new_right, new_bottom]
    return [left, top, right, bottom]


def _round_by_factor(value: float, factor: int) -> int:
    """Round value to nearest multiple of factor"""
    return round(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    """Floor value to nearest multiple of factor"""
    return floor(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    """Ceil value to nearest multiple of factor"""
    return ceil(value / factor) * factor
