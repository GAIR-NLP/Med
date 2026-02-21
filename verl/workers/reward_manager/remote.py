import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.protocol import DataProtoItem
from verl.workers.reward_manager import register

from .remote_proxy import SingleStepRemoteProxyManager


def simple_replace_label(text):
    # Use regex to find 'label': '...' pattern and replace with 'label': 'ui'
    pattern = r"('label':\s*')[^']*(')"
    return re.sub(pattern, r"\1ui\2", text)


def replace_label_with_ui(text):
    # Extract content from <answer> tags
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if not answer_match:
        return text

    json_str = answer_match.group(1)

    try:
        # Parse JSON
        data = json.loads(json_str)

        # Replace label with 'ui'
        for item in data:
            if "label" in item:
                item["label"] = "ui"

        # Reconstruct the string
        new_json_str = json.dumps(data, ensure_ascii=False)
        result = text.replace(answer_match.group(1), new_json_str)
        return result

    except json.JSONDecodeError:
        return text


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


class RewardWorker:
    def __init__(
        self,
        tokenizer,
        reward_server_params,
        is_training,
        step,
        total_steps,
        acc_scale_range: list[float] = [0, 1.0],
        format_scale_range: list[float] = [0, 1.0],
        tool_consistency_scale_range: list[float] = [0, 1.0],
        tool_intrinsic_scale_range: list[float] = [0, 1.0],
        **kwargs,
    ):
        # Tokenizer for decode token
        self.tokenizer = tokenizer
        # Monitor training
        self.is_training = is_training
        self.step = step
        self.total_steps = total_steps

        # acc & format will be scaled by the following parameters
        #   accuracy = acc_scale_reward * accuracy + acc_scale_penalty
        #   format = format_scale_reward * format + format_scale_penalty
        self.acc_scale_range = acc_scale_range
        self.format_scale_range = format_scale_range
        self.tool_consistency_scale_range = tool_consistency_scale_range
        self.tool_intrinsic_scale_range = tool_intrinsic_scale_range
        # Initialize reward server proxy within the actor
        self.reward_server_proxy = SingleStepRemoteProxyManager(
            rm_job=reward_server_params.get("rm_job", "j-f1eycisd8w"),
            rm_num=reward_server_params.get("rm_num", 8),
            rm_port=reward_server_params.get("rm_port", "8192"),
            rm_fun=reward_server_params.get("rm_fun", "/judge"),
        )

    def process_item(self, idx: int, data_item: DataProtoItem):
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

        # fetch the data and response tensor
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode (assoicate with data and response)
        prompt_str: str = self.tokenizer.decode(
            valid_prompt_ids, skip_special_tokens=True
        )  # apply chat template
        response_str: str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        answer: str = data_item.non_tensor_batch["reward_model"]["answer"]  # answer (not formatted)
        ground_truth: str = data_item.non_tensor_batch["reward_model"][
            "ground_truth"
        ]  # solution (formatted)
        query: str = data_item.non_tensor_batch["extra_info"].get(
            "question", ""
        )  # query (without chat template)
        if self.tokenizer.__class__.__name__ == "TokenizersBackend":
            response_str = re.sub(
                r"<\|begin_of_box\|>\\boxed\{([^}]+)\}<\|end_of_box\|>",
                r"\\boxed{\1}",
                response_str,
            )
            response_str = re.sub(
                r"<\|begin_of_box\|>([^<]+)<\|end_of_box\|>", r"\\boxed{\1}", response_str
            )
        if query is None:
            query = ""

        # get reward model info
        data_source: str = data_item.non_tensor_batch["data_source"]
        reward_verifier_style: str = data_item.non_tensor_batch["reward_model"].get("style", "None")
        reward_verifier: str = data_item.non_tensor_batch["reward_model"]["verifier"]

        reward_verifier_parm: dict[str, Any] = data_item.non_tensor_batch["reward_model"].get(
            "verifier_parm", {}
        )
        if reward_verifier_parm is None:
            reward_verifier_parm = {}

        # reward assoicate with image
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

        # prepare reward verifier parm
        reward_verifier_parm["verifier_style"] = reward_verifier_style
        reward_verifier_parm["is_training"] = self.is_training
        reward_verifier_parm["step"] = int(self.step)
        reward_verifier_parm["total_steps"] = int(self.total_steps)
        reward_verifier_parm["image_grid_thw"] = image_grid_thw
        reward_verifier_parm["image_path"] = image_path
        reward_verifier_parm["query"] = query

        # Convert any numpy types to Python native types for JSON serialization
        # serializable_parm = convert_to_serializable(reward_verifier_parm)
        if "ui" in data_source:
            response_str = simple_replace_label(response_str)
        payload = {
            "data_source": data_source,  # data source
            "query": query,  # query (without chat format)
            "prompt": prompt_str,  # add system prompt (chat template) of query
            "answer": answer,  # answer (not formatted)
            "solution": ground_truth,  # solution (formatted)
            "response": response_str,  # response from model
            "reward_verifier": reward_verifier,  # reward verifier
            "reward_verifier_parm": json.dumps(reward_verifier_parm),  # reward verifier parm
        }

        # ========== Apply reward server ==========
        rewards = self.reward_server_proxy.get_reward([payload])
        try:
            gather_rewards = rewards[0]["rewards"]
        except Exception as e:
            print(f"Error in get_reward: {e}", "Payload: ", payload, flush=True)
            gather_rewards = {}
        # ========== End of reward server ==========

        # ========== Apply multi-round / image tools reward ==========
        # assign number of round back into data_item
        data_item.non_tensor_batch["extra_info"]["num_turns"] = data_item.non_tensor_batch.get(
            "__num_turns__", None
        )

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
                        # Aggregate the reward list (using average for now, can be changed to max/sum if needed)
                        tool_successful_rate_dict[tool_name] = sum(reward_list) / len(reward_list)
        # ========== End of multi-round / image tools reward =========

        # ========== Re-org the final reward ==========
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
        # ========== End of format reward ==========

        # ========== Calculate image_crop_and_zoom_in_tool reward ===========
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
        # ========== End of image_crop_and_zoom_in_tool reward ===========

        # ========== Calculate final reward ===========
        gather_rewards["final_reward"] = (
            scaled_accuracy_reward + scaled_format_reward + tool_consistency_reward
        )
        # ========== End of final reward ===========

        result_dict = {
            "id": data_item.non_tensor_batch["extra_info"]["id"],
            "data_source": data_source,
            "prompt": prompt_str,
            "response": response_str,
            "ground_truth": ground_truth,
            "answer": answer,
            "question": query,
            "has_tool_call": has_tool_call,  # used for intrinsic reward computation
            "has_right_tool_call": has_right_tool_call,  # used for intrinsic reward computation
            "uid": data_item.non_tensor_batch.get("uid", "default_group"),  # group identifier
        }
        for reward_key, reward_value in gather_rewards.items():
            result_dict[reward_key] = reward_value

        score = float(gather_rewards["final_reward"])

        return idx, int(valid_response_length), score, result_dict

    def _calculate_intrinsic_reward_for_single_result_TNSprime(self, result):
        """
        Calculate intrinsic reward for a single sample based on stored group statistics.
        Advantage-style version:
        p_call   ≈ (A + eps) / (A + C + 2*eps)
        p_nocall ≈ (D + eps) / (B + D + 2*eps)
        TNS'     = p_call - p_nocall

        Only apply intrinsic reward to A and C (samples that used tools):
        - TNS' > 0: reward A (called tool & correct)
        - TNS' < 0: penalize C (called tool & incorrect)
        B / D intrinsic_reward remains 0
        """
        idx, valid_response_length, _, result_dict = result

        # 1. Check if group statistics exist
        if "group_N_A" not in result_dict:
            return None

        # 2. Get group statistics
        N_A = result_dict["group_N_A"]
        N_B = result_dict["group_N_B"]
        N_C = result_dict["group_N_C"]
        N_D = result_dict["group_N_D"]

        call_total = N_A + N_C  # number of samples that used tools
        nocall_total = N_B + N_D  # number of samples that did not use tools

        # Default values (for logging)
        TNS_prime = 0.0
        p_call = 0.0
        p_nocall = 0.0

        # 3. Compute smoothed p_call / p_nocall (if both sides have samples)
        #    If either side has no samples, keep p_call/p_nocall/TNS' at 0, no shaping
        eps = getattr(self, "tool_intrinsic_smooth_eps", 1.0)
        if call_total > 0 and nocall_total > 0:
            p_call = (N_A + eps) / (call_total + 2 * eps)
            p_nocall = (N_D + eps) / (nocall_total + 2 * eps)
            TNS_prime = p_call - p_nocall

        # 4. Get current sample category
        category = result_dict.get("intrinsic_category", "")

        intrinsic_reward = 0.0

        # Only compute intrinsic reward for A / C samples,
        # and ensure both call / no-call sides have samples (otherwise TNS' is meaningless)
        if category in ["A", "C"] and call_total > 0 and nocall_total > 0:
            # Use scale_range magnitude as the max shaping coefficient
            scale_mag = abs(self.tool_intrinsic_scale_range[1])

            if category == "A" and TNS_prime > 0:
                # Called tool & correct, and tool usage is overall better -> positive reward
                intrinsic_reward = scale_mag * TNS_prime
            elif category == "C" and TNS_prime < 0:
                # Called tool & incorrect, and tool usage is overall worse -> negative penalty
                intrinsic_reward = scale_mag * TNS_prime  # TNS' < 0, result is negative

        # 5. Write back intrinsic reward & statistics (for logging)
        result_dict["tool_intrinsic_reward"] = intrinsic_reward
        result_dict["TNS_prime"] = TNS_prime
        result_dict["p_call"] = p_call
        result_dict["p_nocall"] = p_nocall

        # 6. Recompute final_reward
        result_dict.pop("final_reward", None)
        new_final_reward = sum(
            value
            for key, value in result_dict.items()
            if key.endswith("_reward") and isinstance(value, (int, float))
        )
        result_dict["final_reward"] = new_final_reward

        return (idx, valid_response_length, float(new_final_reward), result_dict)

    def _calculate_intrinsic_reward_for_single_result_v3(self, result):
        """
        Calculate intrinsic reward for a single sample based on stored group statistics.
        """
        idx, valid_response_length, _, result_dict = result

        # Check if group statistics exist
        if "group_N_A" not in result_dict:
            return None

        # Get group statistics
        N_A = result_dict["group_N_A"]
        N_B = result_dict["group_N_B"]
        N_C = result_dict["group_N_C"]
        N_D = result_dict["group_N_D"]

        # Compute TNS
        denominator = N_A + N_B + N_C + N_D
        if denominator == 0:
            denominator = 1e-6  # prevent division by zero
        TNS = (N_B - N_D) / denominator

        # TEQ (Tool Execution Quality): A / (A+C)
        # Reward "reliability" rather than "scarcity"
        teq_denominator = N_A + N_C + 1e-6
        TEQ = N_A / teq_denominator

        # Get current sample category
        category = result_dict.get("intrinsic_category", "")

        # Compute intrinsic reward
        intrinsic_reward = 0.0

        # Only compute intrinsic reward for A and C samples
        if category in ["A", "C"]:
            bonus_int = self.tool_intrinsic_scale_range[1]
            penalty_magnitude = bonus_int
            if category == "A":
                if TNS > 0:  # Category A: called tool & correct - reward when TNS > 0
                    intrinsic_reward = bonus_int * TNS * TEQ
                else:  # Penalize A when TNS < 0
                    intrinsic_reward = penalty_magnitude * TNS
            elif (
                category == "C"
            ):  # Category C: called tool & incorrect - only penalize when TNS < 0
                # penalty = magnitude(0.1) * negative TNS(e.g. -0.5) = -0.05
                intrinsic_reward = penalty_magnitude * min(0, TNS)

        # B and D samples keep intrinsic_reward at 0

        # Add intrinsic reward to result_dict
        result_dict["tool_intrinsic_reward"] = intrinsic_reward

        # Recompute final_reward
        result_dict.pop("final_reward", None)
        new_final_reward = sum(
            value
            for key, value in result_dict.items()
            if key.endswith("_reward") and isinstance(value, (int, float))
        )
        result_dict["final_reward"] = new_final_reward
        result_dict["TNS"] = TNS
        result_dict["TEQ"] = TEQ

        # Return updated tuple
        return (idx, valid_response_length, float(new_final_reward), result_dict)


@register("remote")
class RemoteRewardManager:
    def __init__(
        self, tokenizer, num_examine, compute_score, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs
        self.max_workers = reward_kwargs.get("max_workers", 32)
        self.rm_job = reward_kwargs.get("remote_reward_job_id", "j-xxxxxx")
        self.rm_num = int(reward_kwargs.get("remote_reward_worker_num", "8"))
        self.rm_port = reward_kwargs.get("remote_reward_server_port", "8192")

    def __call__(self, data: DataProto, return_dict: bool = False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]

        results = self.verify(data)
        rewards = []
        already_printed = {}

        all_result_dicts = []
        for idx, length, score, result_dict in results:
            reward_extra_info["result_dicts"].append(result_dict)
            rewards.append(score)
            reward_tensor[idx, length - 1] = score

            data_source = result_dict[self.reward_fn_key]
            if already_printed.get(data_source, 0) < self.num_examine:
                print(result_dict)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

            all_result_dicts.append(result_dict)

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def verify(self, data):
        step, total_steps = data.meta_info["global_steps"], data.meta_info["total_steps"]

        reward_server_params = {"rm_job": self.rm_job, "rm_fun": "/judge"}

        start_time = time.time()
        print(f"=======Start {len(data)} reward, max_workers: {self.max_workers}=======")

        # Create a worker instance
        worker = RewardWorker(
            self.tokenizer,
            reward_server_params,
            step=step,
            total_steps=total_steps,
            **self.reward_kwargs,
        )

        # Use ThreadPoolExecutor for parallel processing
        num_workers = min(self.max_workers, len(data))
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit tasks to the executor
            futures = [
                executor.submit(worker.process_item, i, data_item)
                for i, data_item in enumerate(data)
            ]

            # Gather results as they complete
            for future in as_completed(futures):
                results.append(future.result())

        # Sort results by idx (in-place)
        results.sort(key=lambda x: x[0])

        # ========== Calculate Intrinsic Tool Reward ==========
        # Step 1: Compute group statistics and store in result_dict
        result_dicts = [result_dict for _, _, _, result_dict in results]
        self._add_group_statistics_to_results(result_dicts)

        # Step 2: Use worker instance to compute intrinsic reward
        for i, result in enumerate(results):
            if int(os.getenv("new_instrinsinc_reward_type", 0)):
                updated_result = worker._calculate_intrinsic_reward_for_single_result_TNSprime(
                    result
                )
            else:
                updated_result = worker._calculate_intrinsic_reward_for_single_result_v3(result)
            if updated_result is not None:
                results[i] = updated_result
        # ========== End of Intrinsic Tool Reward ==========

        end_time = time.time()
        print(
            f"=======Complete {len(data)} reward, takes: {round(end_time - start_time, 2)} seconds======="
        )
        return results

    def _add_group_statistics_to_results(self, result_dicts):
        """
        Add group statistics to each sample's result_dict.
        """
        from collections import defaultdict

        # Group by uid and compute statistics
        groups = defaultdict(list)
        group_stats = {}

        for i, result_dict in enumerate(result_dicts):
            uid = result_dict.get("uid", "default_group")
            groups[uid].append((i, result_dict))

        # print(f"Found {len(groups)} groups with sizes: {[len(group_items) for group_items in groups.values()]}")

        # Compute statistics for each group
        for uid, group_items in groups.items():
            N_A = N_B = N_C = N_D = 0

            for result_idx, result_dict in group_items:
                # Get correctness and tool call info
                is_correct = result_dict.get("accuracy_reward", 0) > 0
                has_tool_call = result_dict.get("has_tool_call", False)

                # Classify and record category
                if has_tool_call and is_correct:
                    N_A += 1  # Called tool & correct
                    category = "A"
                elif not has_tool_call and not is_correct:
                    N_B += 1  # No tool & wrong
                    category = "B"
                elif has_tool_call and not is_correct:
                    N_C += 1  # Called tool & wrong
                    category = "C"
                elif not has_tool_call and is_correct:
                    N_D += 1  # No tool & correct
                    category = "D"

                # Store category info in result_dict
                result_dict["intrinsic_category"] = category

            group_stats[uid] = {"N_A": N_A, "N_B": N_B, "N_C": N_C, "N_D": N_D}
            # print(f"Group {uid}: A={N_A}, B={N_B}, C={N_C}, D={N_D}")

        # Add statistics to each sample's result_dict
        for result_dict in result_dicts:
            uid = result_dict.get("uid", "default_group")
            stats = group_stats[uid]

            # Only add group statistics, do not compute intrinsic reward here
            result_dict["group_N_A"] = stats["N_A"]
            result_dict["group_N_B"] = stats["N_B"]
            result_dict["group_N_C"] = stats["N_C"]
            result_dict["group_N_D"] = stats["N_D"]
