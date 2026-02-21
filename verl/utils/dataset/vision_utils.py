# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = Image.open(BytesIO(image["bytes"]))

    return fetch_image(image, image_patch_size=image_patch_size)


VIDEO_FORMAT_HELP = """Currently, we only support the video formats introduced in qwen2-vl.
Refer to https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#using---transformers-to-chat.

eg.
{
    "type": "video",
    "video": [
        "file:///path/to/frame1.jpg",
        "file:///path/to/frame2.jpg"
    ]
}

{
    "type": "video",
    "video": "file:///path/to/video.mp4"
}
# Defaults to fps=2, min_frames=4, max_frames=768

{
    "type": "video",
    "video": "file:///path/to/video.mp4",
    "fps": 2,
    "min_frames": 1,
    "max_frames": 32
}
"""


def process_video(
    video: dict,
    image_patch_size: int = 14,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(
        video,
        image_patch_size=image_patch_size,
        return_video_sample_fps=return_video_sample_fps,
        return_video_metadata=return_video_metadata,
    )


def process_multi_modal_inputs_for_minicpmo(
    input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
):
    # Adjust image bounds based on left padding and cumulative sequence lengths
    # This is necessary for MiniCPM-o's vision-language alignment
    left_padding_length = torch.argmax(attention_mask, dim=1)
    image_bounds = []
    for i in range(len(multi_modal_inputs["image_bound"])):
        image_bound = (
            multi_modal_inputs["image_bound"][i].to(left_padding_length.device)
            - left_padding_length[i]
            + cu_seqlens[i]
        )
        image_bounds.append(image_bound)

    # Flatten pixel values list for MiniCPM-o processing
    pixel_values = []
    for i in range(len(multi_modal_inputs["pixel_values"])):
        pixel_values.extend([p for p in multi_modal_inputs["pixel_values"][i]])

    multi_modal_inputs["pixel_values"] = [pixel_values]
    multi_modal_inputs["image_bound"] = [torch.vstack(image_bounds)]
    multi_modal_inputs["tgt_sizes"] = [torch.vstack(multi_modal_inputs["tgt_sizes"])]
    multi_modal_inputs["input_ids"] = input_ids
    multi_modal_inputs["attention_mask"] = attention_mask
    multi_modal_inputs["position_ids"] = position_ids
    return {"data": multi_modal_inputs}


def random_square_block_mask(
    image: Image.Image,
    mask_ratio_range: list[float] = [0, 0.05, 0.10, 0.15, 0.2],
    block_size_ratio: float = 0.1,
) -> Image.Image:
    """
    Applies a random square block mask to the input PIL image.

    Args:
        image (Image.Image): The input PIL.Image object.
        mask_ratio (float): The desired ratio of the total masked area to the
                            total image area. Defaults to 0.1 (10%).
        block_size_ratio (float): The ratio used to determine the block size.
                                  The side length of the square block will be
                                  `min(image_height, image_width) * block_size_ratio`.
                                  Defaults to 0.1 (10%).

    Returns:
        Image.Image: A new PIL.Image object with the random square block mask applied.
    """
    # --- 1. Convert the image to a NumPy array ---
    img_array = np.array(image.copy())

    # --- 2. Get image dimensions and calculate block size ---
    h, w, _ = img_array.shape

    mask_ratio = random.choice(mask_ratio_range)
    # Calculate the side length of the block (as an integer)
    block_size = int(min(h, w) * block_size_ratio)

    # If the calculated block size is less than 1 pixel, masking is not possible.
    # Return a copy of the original image.
    if block_size < 1:
        return image.copy()

    # --- 3. Calculate the number of blocks to add ---
    # Target mask area
    target_mask_area = h * w * mask_ratio
    # Area of a single block
    block_area = block_size * block_size
    # Number of blocks to add (use math.ceil to ensure the mask_ratio is met or exceeded)
    num_blocks_to_add = math.ceil(target_mask_area / block_area)

    # --- 4. Loop to add mask blocks ---
    for _ in range(num_blocks_to_add):
        # Randomly determine the top-left corner of the block,
        # ensuring the block is placed within the image boundaries.
        top = np.random.randint(0, h - block_size + 1)
        left = np.random.randint(0, w - block_size + 1)

        # Set the square region to black
        img_array[top : top + block_size, left : left + block_size, :] = 0

    # --- 5. Convert the array back to an image and return ---
    return Image.fromarray(img_array)
