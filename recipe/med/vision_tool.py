import json
import logging
import threading
from collections.abc import Callable
from contextlib import ExitStack
from enum import Enum
from typing import Any
from uuid import uuid4

import ray
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .vision_tool_utils import (
    decode_base64_to_image,
    encode_base64_image,
    perform_single_vision_tool,
)

logger = logging.getLogger(__name__)


TOOL_REWARD = 1.0
TOOL_PENALTY = 0.0


class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})  # type: ignore
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisionExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = (
            self._init_rate_limit(rate_limit) if enable_global_rate_limit else None
        )

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(  # type: ignore
            name="rate-limiter", get_if_exists=True
        ).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable, *fn_args, **fn_kwargs):
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                return fn(*fn_args, **fn_kwargs)
        else:
            return fn(*fn_args, **fn_kwargs)


def init_vision_execution_pool(
    num_workers: int,
    enable_global_rate_limit=True,
    rate_limit=10,
    mode: PoolMode = PoolMode.ThreadMode,
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisionExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(
                enable_global_rate_limit=enable_global_rate_limit,
                rate_limit=rate_limit,  # type: ignore
            )
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageCropAndZoomInTool(BaseTool):
    """
    A tool for cropping and zooming into specific regions of images with multi-image support.
    This class adheres to the provided BaseTool structure.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initializes the tool instance. The schema is expected to be provided by the framework.

        Args:
            config: A dictionary containing configuration for the tool.
            tool_schema: The OpenAI function tool schema provided by the instantiating framework.
        """
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.tool_reward = config.get("reward", TOOL_REWARD)
        self.tool_penalty = config.get("penalty", TOOL_PENALTY)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_vision_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Vision service configuration
        self.vision_service_url = config.get("vision_service_url")
        assert self.vision_service_url, "Configuration must include 'vision_service_url'"

    @staticmethod
    def get_default_schema() -> "OpenAIFunctionToolSchema":
        """
        Returns the default schema for this tool in a readable dictionary format,
        which is then validated into the Pydantic model.
        """
        schema_dict = {
            "type": "function",
            "function": {
                "name": "image_crop_and_zoom_in_tool",
                "description": "Crops and zooms into a specific region of an image based on provided relative coordinates. All coordinate values should be float numbers with the range [0,1000.0].",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": "Defines the rectangular area for cropping in the format [x1, y1, x2, y2]. (x1, y1) marks the upper-left corner, and (x2, y2) marks the lower-right corner. All coordinates must be in the range [0, 1000] with x1 < x2 and y1 < y2.",
                        },
                        "label": {
                            "type": "string",
                            "description": "A descriptive name for the object within the cropped region.",
                        },
                        "image_index": {
                            "type": "number",
                            "description": "The index of the image to apply the crop and zoom operation on, starting from 0.",
                        },
                    },
                    "required": ["bbox_2d", "label", "image_index"],
                    "strict": True,
                },
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    async def create(
        self,
        create_kwargs: dict[str, Any],
        instance_id: str | None = None,
    ) -> tuple[str, ToolResponse]:
        """
        Initializes the tool state for a trajectory.

        Args:
            instance_id: An optional unique identifier for the instance.
            images: List of images for this trajectory.

        Returns:
            The unique instance ID for this trajectory.
        """
        image = create_kwargs.get("image", [])
        raw_query = create_kwargs.get("raw_query", "")

        if isinstance(image, list):
            images = image
        else:
            images = [image]

        if not images:
            raise ValueError("A list of `images` must be provided when creating an instance.")

        # Convert PIL images to base64 if needed
        processed_images = []
        for image in images:
            if isinstance(image, Image.Image):
                processed_images.append(encode_base64_image(image))
            else:
                processed_images.append(image)

        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {
            "images": processed_images,
            "operation_history": [],
            "cumulative_reward": 0.0,
            "tool_call_count": 0,
            "raw_query": raw_query,
            **create_kwargs,
        }
        return instance_id, ToolResponse()

    def execute_vision_tool(
        self,
        image_b64: str,
        parameters: dict[str, Any],
        vision_service_url: str,
        timeout: int,
    ) -> tuple[str | None, dict]:
        # Convert new schema parameters to backend-compatible format
        bbox_2d = parameters.get("bbox_2d")
        transformed_params = {**parameters}

        if bbox_2d:
            # Backend expects 'crop_box' parameter
            transformed_params["crop_box"] = bbox_2d

        # Only pass parameters that the backend service expects
        # label and image_index are for tool logic, not backend service

        result_image_b64, metadata = perform_single_vision_tool(
            vision_tool_service_url=vision_service_url,
            image_b64=image_b64,
            parameters=transformed_params,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )

        # Let the caller handle None result, don't return original image here
        return result_image_b64, metadata

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Executes the core function of the tool.

        Args:
            instance_id: The ID of the tool instance to execute on.
            parameters: A dictionary of parameters matching the tool's schema.

        Returns:
            A tuple containing (observation, tool_reward, metrics).
        """
        if instance_id not in self._instance_dict:
            raise KeyError(f"Instance ID '{instance_id}' not found. Please call 'create' first.")

        timeout = self.timeout
        instance = self._instance_dict[instance_id]
        images = instance["images"]

        # Validate bbox_2d parameter
        bbox_2d = parameters.get("bbox_2d")
        if isinstance(bbox_2d, str):
            try:
                bbox_2d = json.loads(bbox_2d)
            except:
                pass
        if not isinstance(bbox_2d, (tuple, list)) or len(bbox_2d) != 4:
            tool_response = ToolResponse(
                text=f"Error: bbox_2d must be a list containing 4 numbers. But You have {bbox_2d}."
            )
            metadata = {
                "message": f"bbox_2d must be a list containing 4 numbers. But You have {bbox_2d}.",
                "status": "error",
                "tool_reward": False,
            }
            return tool_response, self.tool_penalty, metadata

        # Validate image_index parameter
        image_index = parameters.get("image_index", 0)
        if isinstance(image_index, str):
            try:
                image_index = int(image_index)
            except:
                pass
        if (
            not isinstance(image_index, (int, float))
            or image_index < 0
            or image_index >= len(images)
        ):
            tool_response = ToolResponse(
                text=f"Error: image_index must be a valid index (0 to {len(images)-1}). But You have {image_index}."
            )
            metadata = {
                "message": f"image_index must be a valid index (0 to {len(images)-1}). But You have {image_index}.",
                "status": "error",
                "tool_reward": False,
            }
            return tool_response, self.tool_penalty, metadata

        # Get the target image from the images list
        target_image = images[int(image_index)]

        # Execute search using Ray execution pool
        processed_image_b64, metadata = await self.execution_pool.execute.remote(
            self.execute_vision_tool,
            target_image,
            parameters,
            self.vision_service_url,
            timeout,
        )

        # Handle vision tool execution errors
        if processed_image_b64 is None:
            tool_response = ToolResponse(text=f"Error: {metadata.get('message')}")
            metadata["tool_reward"] = False
            return tool_response, self.tool_penalty, metadata

        tool_reward = self.tool_reward if metadata["tool_reward"] else self.tool_penalty

        # Append the processed image to the images list (key change!)
        instance["images"].append(processed_image_b64)
        instance["operation_history"].append({"params": parameters, "reward": tool_reward})
        instance["tool_call_count"] += 1
        instance["cumulative_reward"] += tool_reward
        instance["cumulative_reward"] = max(
            self.tool_penalty, min(instance["cumulative_reward"], self.tool_reward)
        )

        processed_image = decode_base64_to_image(processed_image_b64)

        tool_response = ToolResponse(
            image=[processed_image],
        )
        return tool_response, tool_reward, metadata

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculates the cumulative reward for the instance.

        Args:
            instance_id: The ID of the tool instance.

        Returns:
            The total cumulative reward for the instance.
        """
        if instance_id not in self._instance_dict:
            return 0.0

        reward = self._instance_dict[instance_id]["cumulative_reward"]
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Cleans up all resources allocated for an instance.

        Args:
            instance_id: The ID of the tool instance to release.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class VisionTool(BaseTool):
    """
    A tool for cropping and zooming images, designed for multi-turn agent interactions.
    This class adheres to the provided BaseTool structure.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Initializes the tool instance. The schema is expected to be provided by the framework.

        Args:
            config: A dictionary containing configuration for the tool.
            tool_schema: The OpenAI function tool schema provided by the instantiating framework.
        """
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 120)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)
        self.tool_reward = TOOL_REWARD
        self.tool_penalty = TOOL_PENALTY

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_vision_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Vision service configuration
        self.vision_service_url = config.get("vision_service_url")
        assert self.vision_service_url, "Configuration must include 'vision_service_url'"

    @staticmethod
    def get_default_schema() -> "OpenAIFunctionToolSchema":
        """
        Returns the default schema for this tool in a readable dictionary format,
        which is then validated into the Pydantic model.
        """
        schema_dict = {
            "type": "function",
            "function": {
                "name": "crop_and_zoom",
                "description": "Crops a specified region of an image using relative coordinates. All coordinate values must be integers within the range [0, 1000].",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "crop_box": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            "description": "A tuple [x1, y1, x2, y2] defining the crop area using relative coordinates, where all values must be within the range [0, 1000]. (x1, y1) represents the coordinates of the top-left corner, and (x2, y2) represents the coordinates of the bottom-right corner. The condition x1 < x2 and y1 < y2 must be satisfied.",
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "zoom_factor": {
                            "type": "number",
                            "description": "Optional zoom factor. If omitted, it will be auto-calculated to fit the cropped area to the original image's dimensions.",
                        },
                        "output_resolution_limit": {
                            "type": "integer",
                            "description": "Optional safety cap for the output image's longest side. Defaults to 1024.",
                        },
                    },
                    "required": ["crop_box"],
                    "strict": True,
                },
            },
        }
        return OpenAIFunctionToolSchema.model_validate(schema_dict)

    async def create(
        self,
        create_kwargs: dict[str, Any],
        instance_id: str | None = None,
    ) -> tuple[str, ToolResponse]:
        """
        Initializes the tool state for a trajectory.

        Args:
            instance_id: An optional unique identifier for the instance.
            original_image: The starting image for this trajectory.

        Returns:
            The unique instance ID for this trajectory.
        """
        image = create_kwargs.get("image", None)
        raw_query = create_kwargs.get("raw_query", "")

        if image is None:
            raise ValueError("An `image` must be provided when creating an instance.")
        elif isinstance(image, Image.Image):
            image = encode_base64_image(image)

        instance_id = instance_id or str(uuid4())
        self._instance_dict[instance_id] = {
            "original_image": image,
            "current_image": image,
            "operation_history": [],
            "cumulative_reward": 0.0,
            "tool_call_count": 0,
            "raw_query": raw_query,
            **create_kwargs,
        }
        return instance_id, ToolResponse()

    def execute_vision_tool(
        self,
        image_b64: str,
        parameters: dict[str, Any],
        vision_service_url: str,
        timeout: int,
    ) -> tuple[str, dict]:
        result_image_b64, metadata = perform_single_vision_tool(
            vision_tool_service_url=vision_service_url,
            image_b64=image_b64,
            parameters=parameters,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        # we should always expect this since we don't have correct answer
        if result_image_b64 is None:
            logger.debug("Error in Vision Tool, return raw image")
            result_image_b64 = image_b64

        return result_image_b64, metadata

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """
        Executes the core function of the tool.

        Args:
            instance_id: The ID of the tool instance to execute on.
            parameters: A dictionary of parameters matching the tool's schema.

        Returns:
            A tuple containing (observation, tool_reward, metrics).
        """
        if instance_id not in self._instance_dict:
            raise KeyError(f"Instance ID '{instance_id}' not found. Please call 'create' first.")

        timeout = self.timeout
        instance = self._instance_dict[instance_id]
        original_image = instance["original_image"]
        raw_query = instance.get("raw_query")

        crop_box = parameters.get("crop_box")
        if not isinstance(crop_box, (tuple, list)) or len(crop_box) != 4:
            tool_response = ToolResponse(
                text=f"Tool Call: crop_and_zoom with parameters {parameters}\n\n"
                f"Status: error\n\n"
                f"Message: 'crop_box' must be a list containing 4 numbers. But You have {crop_box}. \n\n"
                f"Please continue to answer the original question: {raw_query}"
            )
            metadata = {
                "message": f"crop_box must be a list containing 4 numbers. But You have {crop_box}.",
                "status": "error",
                "tool_reward": self.tool_penalty,
            }

            return tool_response, self.tool_penalty, metadata

        # Execute search using Ray execution pool
        processed_image_b64, metadata = await self.execution_pool.execute.remote(
            self.execute_vision_tool,
            original_image,
            parameters,
            self.vision_service_url,
            timeout,
        )

        tool_reward = self.tool_reward if metadata["tool_reward"] else self.tool_penalty

        instance["current_image"] = processed_image_b64
        instance["operation_history"].append({"params": parameters, "reward": tool_reward})
        instance["tool_call_count"] += 1
        instance["cumulative_reward"] += tool_reward
        instance["cumulative_reward"] = max(
            self.tool_penalty, min(instance["cumulative_reward"], self.tool_reward)
        )

        processed_image = decode_base64_to_image(processed_image_b64)

        tool_response = ToolResponse(
            text=f"Tool Call: crop_and_zoom with parameters {parameters}\n\n"
            f"Status: {metadata['status']}\n\n"
            f"Message: {metadata['message']}\n\n"
            f"Please continue to answer the original question: {raw_query}",
            image=[processed_image],
        )
        return tool_response, tool_reward, metadata

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """
        Calculates the cumulative reward for the instance.

        Args:
            instance_id: The ID of the tool instance.

        Returns:
            The total cumulative reward for the instance.
        """
        if instance_id not in self._instance_dict:
            return 0.0

        reward = self._instance_dict[instance_id]["cumulative_reward"]
        return reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """
        Cleans up all resources allocated for an instance.

        Args:
            instance_id: The ID of the tool instance to release.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
