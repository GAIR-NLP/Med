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

import asyncio
import json
import os
from typing import Any

import numpy as np

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase


def scale_value_from_01(value: float, target_range: list[float]) -> float:
    """
    Scales a value from the [0, 1] interval to a new specified range.

    Args:
    value: The original value, which should be in the range [0, 1].
    target_range: A tuple of two floats (min, max) defining the lower and upper
                    bounds of the target range.

    Returns:
    The scaled float value.
    """
    # Clamp value to [0, 1] range instead of asserting
    value = max(0.0, min(1.0, value))
    assert (
        len(target_range) == 2
    ), f"target range shoud only have two values, but found {target_range}"

    lower_bound, upper_bound = target_range

    return lower_bound + value * (upper_bound - lower_bound)


class AsyncRemoteProxyManager:
    """Async version of SingleStepRemoteProxyManager for remote reward server calls."""

    def __init__(self, rm_job, rm_num, rm_port, rm_fun):
        self.rm_job = rm_job
        self.rm_num = rm_num
        self.rm_port = rm_port
        self.rm_fun = rm_fun

    async def get_reward(self, payloads):
        """Async wrapper for remote reward server call.

        Note: This is a placeholder implementation. In a real scenario,
        you would replace this with actual async HTTP calls to your reward server.
        """
        # Import the actual remote proxy for synchronous call
        try:
            from verl.workers.reward_manager.remote_proxy import SingleStepRemoteProxyManager

            # Create synchronous proxy
            sync_proxy = SingleStepRemoteProxyManager(
                rm_job=self.rm_job, rm_num=self.rm_num, rm_port=self.rm_port, rm_fun=self.rm_fun
            )

            # Run the synchronous call in executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            rewards = await loop.run_in_executor(None, sync_proxy.get_reward, payloads)
            return rewards

        except ImportError:
            # Fallback implementation if remote_proxy is not available
            print("Warning: remote_proxy not available, using mock rewards")
            return [{"rewards": {"format_reward": 1.0, "accuracy_reward": 0.0}} for _ in payloads]


@register("remote")
class RemoteRewardLoopManager(RewardLoopManagerBase):
    """Async remote reward manager that maintains all original logic."""

    def __init__(
        self,
        config,
        tokenizer,
        is_training=True,
        acc_scale_range: list[float] = [0, 1.0],
        format_scale_range: list[float] = [0, 1.0],
        tool_consistency_scale_range: list[float] = [0, 1.0],
        tool_intrinsic_scale_range: list[float] = [0, 1.0],
        reward_server_params=None,
    ):
        super().__init__(config, tokenizer)

        # Store configuration
        self.is_training = is_training
        self.acc_scale_range = acc_scale_range
        self.format_scale_range = format_scale_range
        self.tool_consistency_scale_range = tool_consistency_scale_range
        self.tool_intrinsic_scale_range = tool_intrinsic_scale_range

        # Initialize reward server parameters
        if reward_server_params is None:
            reward_server_params = {
                "rm_job": os.environ.get("_REMOTE_REWARD_JOB_ID", "j-f1eycisd8w"),
                "rm_num": int(os.environ.get("_REMOTE_REWARD_WORKER_NUM", "8")),
                "rm_port": os.environ.get("_REMOTE_REWARD_SERVER_PORT", "8192"),
                "rm_fun": "/judge",
            }

        # Initialize async reward server proxy
        self.reward_server_proxy = AsyncRemoteProxyManager(
            rm_job=reward_server_params.get("rm_job", "j-f1eycisd8w"),
            rm_num=reward_server_params.get("rm_num", 8),
            rm_port=reward_server_params.get("rm_port", "8192"),
            rm_fun=reward_server_params.get("rm_fun", "/judge"),
        )

    async def run_single(self, data: DataProto) -> dict:
        """Process a single data item and return reward information.

        This method contains all the original reward calculation logic from
        RewardWorker.process_item, adapted for async execution.
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]

        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            else:
                return obj

        # Extract data and response tensor (same logic as original)
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode text (run in executor to avoid blocking)
        prompt_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        )
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )

        # Extract metadata (same logic as original)
        answer: str = data_item.non_tensor_batch["reward_model"]["answer"]
        ground_truth: str = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        query: str = data_item.non_tensor_batch["extra_info"].get("question", "")

        # Get reward model info (same logic as original)
        data_source: str = data_item.non_tensor_batch["data_source"]
        reward_verifier_style: str = data_item.non_tensor_batch["reward_model"]["style"]
        reward_verifier: str = data_item.non_tensor_batch["reward_model"]["verifier"]

        reward_verifier_parm: dict[str, Any] = data_item.non_tensor_batch["reward_model"].get(
            "verifier_parm", {}
        )
        if reward_verifier_parm is None:
            reward_verifier_parm = {}

        # Handle image data (same logic as original)
        image_grid_thw = None
        if (
            "multi_modal_inputs" in data_item.non_tensor_batch
            and "image_grid_thw" in data_item.non_tensor_batch["multi_modal_inputs"]
        ):
            image_grid_thw = data_item.non_tensor_batch["multi_modal_inputs"][
                "image_grid_thw"
            ].numpy()
            # TODO: 14 is the patch size of ViT, dynamically adjust it
            image_grid_thw = [(int(t), int(h * 14), int(w * 14)) for t, h, w in image_grid_thw]

        image_path: list[str] | None = data_item.non_tensor_batch["extra_info"].get(
            "image_path", None
        )
        if isinstance(image_path, str):
            image_path = [image_path]

        # Prepare reward verifier parameters (same logic as original)
        # Get step and total_steps from meta_info instead of instance variables
        step = data.meta_info.get("global_steps", 0)
        total_steps = data.meta_info.get("total_steps", 1)

        reward_verifier_parm["verifier_style"] = reward_verifier_style
        reward_verifier_parm["is_training"] = self.is_training
        reward_verifier_parm["step"] = int(step)
        reward_verifier_parm["total_steps"] = int(total_steps)
        reward_verifier_parm["image_grid_thw"] = image_grid_thw
        reward_verifier_parm["image_path"] = image_path
        reward_verifier_parm["query"] = query

        # Prepare payload for reward server (same logic as original)
        payload = {
            "data_source": data_source,
            "query": query,
            "prompt": prompt_str,
            "answer": answer,
            "solution": ground_truth,
            "response": response_str,
            "reward_verifier": reward_verifier,
            "reward_verifier_parm": json.dumps(reward_verifier_parm),
        }

        # Call reward server asynchronously
        try:
            rewards = await self.reward_server_proxy.get_reward([payload])
            gather_rewards = rewards[0]["rewards"]
        except Exception as e:
            print(f"Error in get_reward: {e}", "Payload: ", payload, flush=True)
            gather_rewards = {"format_reward": 0.0, "accuracy_reward": 0.0}

        # Process multi-round / image tools reward (same logic as original)
        tools_kwargs = data_item.non_tensor_batch["extra_info"].get("tools_kwargs", {})
        need_tools_kwargs = data_item.non_tensor_batch["extra_info"].get("need_tools_kwargs", False)

        # Extract tool rewards from tool_metrics and compute successful rates
        tool_successful_rate_dict = {}
        if need_tools_kwargs:
            tool_metrics = data_item.non_tensor_batch.get("tool_metrics", {})
            for tool_name, metrics_list in tool_metrics.items():
                if tool_name in tools_kwargs and isinstance(metrics_list, list):
                    # Extract all tool_reward values from the metrics list
                    reward_list = [
                        metric["tool_reward"]
                        for metric in metrics_list
                        if isinstance(metric, dict) and "tool_reward" in metric
                    ]
                    if reward_list:
                        # Aggregate the reward list (using average for now)
                        tool_successful_rate_dict[tool_name] = sum(reward_list) / len(reward_list)

        # Apply scaling to rewards (same logic as original)
        format_reward_01 = gather_rewards["format_reward"]
        accuracy_reward_01 = gather_rewards["accuracy_reward"]

        if self.is_training:
            scaled_format_reward = scale_value_from_01(format_reward_01, self.format_scale_range)
            scaled_accuracy_reward = scale_value_from_01(accuracy_reward_01, self.acc_scale_range)
            gather_rewards["format_reward"] = scaled_format_reward
            gather_rewards["accuracy_reward"] = scaled_accuracy_reward
        else:
            scaled_format_reward = format_reward_01
            scaled_accuracy_reward = accuracy_reward_01

        # Calculate image_crop_and_zoom_in_tool reward (same logic as original)
        if need_tools_kwargs:
            qwen3vl_tool_call_tag = data_item.non_tensor_batch["extra_info"].get(
                "qwen3vl_call_tool_discriminative", False
            )
            tool_name = "image_crop_and_zoom_in_tool"
            has_right_tool_call = (
                tool_name in tool_successful_rate_dict and tool_successful_rate_dict[tool_name] > 0
            )
            has_tool_call = tool_name in tool_successful_rate_dict

            # Check if tool call behavior matches the expected tag
            behavior_matches = (qwen3vl_tool_call_tag and has_right_tool_call) or (
                not qwen3vl_tool_call_tag and not has_tool_call
            )

            if accuracy_reward_01:
                # If correct, give bonus when behavior matches expectation
                tool_consistency_reward = (
                    self.tool_consistency_scale_range[-1] if behavior_matches else 0.0
                )
            else:
                # If incorrect, give penalty when behavior doesn't match expectation
                tool_consistency_reward = (
                    0 if behavior_matches else self.tool_consistency_scale_range[0]
                )
        else:
            tool_consistency_reward = 0.0
            has_right_tool_call = False
            has_tool_call = False

        gather_rewards["tool_consistency_reward"] = tool_consistency_reward

        # Calculate final reward (same logic as original)
        gather_rewards["final_reward"] = (
            scaled_accuracy_reward + scaled_format_reward + tool_consistency_reward
        )

        # Prepare result dictionary (same logic as original)
        result_dict = {
            "id": data_item.non_tensor_batch["extra_info"]["id"],
            "data_source": data_source,
            "prompt": prompt_str,
            "response": response_str,
            "ground_truth": ground_truth,
            "answer": answer,
            "question": query,
            "has_tool_call": has_tool_call,
            "has_right_tool_call": has_right_tool_call,
            "uid": data_item.non_tensor_batch.get("uid", "default_group"),
        }

        # Add all reward values to result_dict
        for reward_key, reward_value in gather_rewards.items():
            result_dict[reward_key] = reward_value

        reward_score = float(gather_rewards["final_reward"])

        return {
            "reward_score": reward_score,
            "reward_extra_info": result_dict,
        }
