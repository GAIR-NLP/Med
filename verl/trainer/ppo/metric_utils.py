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
"""
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import torch

import wandb
from verl import DataProto
from verl.utils.import_utils import deprecated


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(
    batch: DataProto, result_dicts: list[dict[str, Any]], step: int, use_critic: bool = True
) -> dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
            - num_turns/mean, max, min: Statistics about the number of multi-turn conversations
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["response_mask"].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    is_correct = [d["accuracy_reward"] == 1.0 for d in result_dicts]
    correction_mask = torch.tensor(is_correct, dtype=torch.bool)
    correct_response_length = torch.masked_select(response_length, correction_mask)
    wrong_response_length = torch.masked_select(response_length, ~correction_mask)

    aborted_mask = (response_length == 0).bool()
    non_aborted_mask = ~aborted_mask

    non_aborted_sequence_score = sequence_score[non_aborted_mask]
    non_aborted_sequence_reward = sequence_reward[non_aborted_mask]

    score_mean = torch.mean(non_aborted_sequence_score).detach().item()

    reward_mean = torch.mean(non_aborted_sequence_reward).detach().item()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    reward_values = {}
    for result in result_dicts:
        for key, value in result.items():
            if "reward" in key and value is not None:
                reward_values.setdefault(key, []).append(value)

    rew_metric_dict = {}
    for reward_name, values in reward_values.items():
        key = f"rews/{reward_name}"
        rew_metric_dict[key] = np.mean(values)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # Aborted samples and non-aborted response length statistics
    # response_length_non_aborted/*: statistics computed on non-aborted samples only
    aborted_ratio = torch.mean(aborted_mask.float()).detach().item()

    non_aborted_response_length = response_length[non_aborted_mask]
    if non_aborted_response_length.numel() > 0:
        non_aborted_response_length_mean = torch.mean(non_aborted_response_length).detach().item()
        non_aborted_response_length_max = torch.max(non_aborted_response_length).detach().item()
        non_aborted_response_length_min = torch.min(non_aborted_response_length).detach().item()
        non_aborted_response_length_clip_ratio = (
            torch.mean(torch.eq(non_aborted_response_length, max_response_length).float())
            .detach()
            .item()
        )
    else:
        raise ValueError("All samples are aborted, this should not happen.")

    metrics = {
        # score
        "rews/total_score": score_mean,
        # reward
        "rews/total_rew": reward_mean,
        # adv
        "rews/advantage": torch.mean(valid_adv).detach().item(),
        # returns
        "rews/return": torch.mean(valid_returns).detach().item(),
        # separate rews
        **rew_metric_dict,
        **(
            {
                # values
                "training/vf_value": torch.mean(valid_values).detach().item(),
                # vf explained var
                "training/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5))
                .detach()
                .item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean_length": torch.mean(response_length).detach().item(),
        "response_length/length_clip_ratio": torch.mean(
            torch.eq(response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # correct response length
        "response_length/mean_correct_length": torch.mean(correct_response_length).detach().item(),
        "response_length/length_correct_clip_ratio": torch.mean(
            torch.eq(correct_response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # wrong response length
        "response_length/mean_wrong_length": torch.mean(wrong_response_length).detach().item(),
        "response_length/length_wrong_clip_ratio": torch.mean(
            torch.eq(wrong_response_length, max_response_length).float()
        )
        .detach()
        .item(),
        # response length (non-aborted only)
        # These statistics exclude aborted samples to avoid skew from zeros
        "response_length_non_aborted/mean": non_aborted_response_length_mean,
        "response_length_non_aborted/max": non_aborted_response_length_max,
        "response_length_non_aborted/min": non_aborted_response_length_min,
        "response_length_non_aborted/clip_ratio": non_aborted_response_length_clip_ratio,
        # aborted ratio
        # Fraction of samples whose response length is zero
        "response/aborted_ratio": aborted_ratio,
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float())
        .detach()
        .item(),
    }

    by_source, idx_by_source = defaultdict(list), defaultdict(list)
    for i, d in enumerate(result_dicts):
        data_source = d["data_source"].replace("/", "_")
        by_source[data_source].append(d)
        idx_by_source[data_source].append(i)

    data_source_metrics = {}
    for data_source in idx_by_source:
        idxs = idx_by_source[data_source]
        data_source_result_dicts = by_source[data_source]

        data_source_reward_values = {}
        for result in data_source_result_dicts:
            for key, value in result.items():
                if "reward" in key and value is not None:
                    data_source_reward_values.setdefault(key, []).append(value)

        data_source_rew_metric_dict = {}
        for reward_name, values in data_source_reward_values.items():
            key = f"rews_{reward_name}/{data_source}"
            data_source_rew_metric_dict[key] = np.mean(values)

        data_source_response_length = response_length[idxs]

        data_source_prompt_texts = [d["prompt"] for d in data_source_result_dicts]
        data_source_response_texts = [d["response"] for d in data_source_result_dicts]
        data_source_is_correct = [d["accuracy_reward"] == 1.0 for d in data_source_result_dicts]
        data_source_ids = [d["id"] for d in data_source_result_dicts]

        words = [
            # 英文词
            "re-check",
            "re-evaluate",
            "re-examine",
            "re-think",
            "recheck",
            "reevaluate",
            "reexamine",
            "reevaluation",
            "rethink",
            "check again",
            "think again",
            "try again",
            "verify",
            "wait",
            # 中文词
            "重新检查",
            "重新评估",
            "重新审视",
            "重新思考",
            "再检查",
            "再思考",
            "再试一次",
            "验证",
            "等待",
            "评审",
            "校验",
            "反思",
            "调整",
            "修正",
        ]
        reflection_metrics = _compute_words_metrics_and_tables(
            words=words,
            ids=data_source_ids,
            prompt_texts=data_source_prompt_texts,
            response_texts=data_source_response_texts,
            is_correct=data_source_is_correct,
            step=step,
            metric_name="reflection",
            data_source=data_source,
            upload_example_table=False,
        )

        tool_call_words = ["<tool_response>"]
        tool_call_metrics = _compute_words_metrics_and_tables(
            words=tool_call_words,
            ids=data_source_ids,
            prompt_texts=data_source_prompt_texts,
            response_texts=data_source_response_texts,
            is_correct=data_source_is_correct,
            step=step,
            metric_name="tool_call",
            data_source=data_source,
            upload_example_table=False,
        )
        tool_reward_metrics = _compute_tool_status(result_dicts)

        data_source_correction_mask = torch.tensor(data_source_is_correct, dtype=torch.bool)
        data_source_correct_response_length = torch.masked_select(
            data_source_response_length, data_source_correction_mask
        )
        data_source_wrong_response_length = torch.masked_select(
            data_source_response_length, ~data_source_correction_mask
        )

        data_source_metrics.update(
            {
                # response length
                f"response_length_{data_source}/mean_length": torch.mean(
                    data_source_response_length
                )
                .detach()
                .item(),
                f"response_length_{data_source}/length_clip_ratio": torch.mean(
                    torch.eq(data_source_response_length, max_response_length).float()
                )
                .detach()
                .item(),
                # correct response length
                f"response_length_{data_source}/mean_correct_length": torch.mean(
                    data_source_correct_response_length
                )
                .detach()
                .item(),
                f"response_length_{data_source}/length_correct_clip_ratio": torch.mean(
                    torch.eq(data_source_correct_response_length, max_response_length).float()
                )
                .detach()
                .item(),
                # wrong response length
                f"response_length_{data_source}/mean_wrong_length": torch.mean(
                    data_source_wrong_response_length
                )
                .detach()
                .item(),
                f"response_length_{data_source}/length_wrong_clip_ratio": torch.mean(
                    torch.eq(data_source_wrong_response_length, max_response_length).float()
                )
                .detach()
                .item(),
                **reflection_metrics,
                **tool_call_metrics,
                **tool_reward_metrics,
                **data_source_rew_metric_dict,
            }
        )

    metrics.update(data_source_metrics)

    # multi-turn conversation
    if "__num_turns__" in batch.non_tensor_batch:
        num_turns = batch.non_tensor_batch["__num_turns__"]
        metrics["num_turns/min"] = num_turns.min()
        metrics["num_turns/max"] = num_turns.max()
        metrics["num_turns/mean"] = num_turns.mean()

    if "tool_call_counts" in batch.non_tensor_batch:
        tool_call_counts = batch.non_tensor_batch["tool_call_counts"]
        metrics["tool_call_counts/min"] = tool_call_counts.min()
        metrics["tool_call_counts/max"] = tool_call_counts.max()
        metrics["tool_call_counts/mean"] = tool_call_counts.mean()

    if "tool_metrics" in batch.non_tensor_batch:
        tool_status_cnt = defaultdict(lambda: {"success": 0, "warning": 0, "error": 0})
        tool_status_ratio = {}
        for tool_metrics in batch.non_tensor_batch["tool_metrics"]:
            for tool_name, tool_calls in tool_metrics.items():
                for tool_call in tool_calls:
                    if "status" in tool_call:
                        tool_status_cnt[tool_name][tool_call["status"]] += 1
                tool_cnt = sum(tool_status_cnt[tool_name].values())
                tool_status_ratio[tool_name] = {
                    status: cnt / tool_cnt for status, cnt in tool_status_cnt[tool_name].items()
                }

            if tool_metrics and tool_name in tool_status_ratio:
                metrics[f"tool_status/{tool_name}_ratio"] = tool_status_ratio[tool_name]
            if tool_metrics and tool_name in tool_status_cnt:
                metrics[f"tool_status/{tool_name}_cnt"] = tool_status_cnt[tool_name]

    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing metrics
    (in milliseconds) for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(
    batch: DataProto, timing_raw: dict[str, float], n_gpus: int
) -> dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def _compute_words_metrics_and_tables(
    words: list[str],
    ids: list[int],
    prompt_texts: list[str],
    response_texts: list[str],
    is_correct: list[bool],
    step: int,
    metric_name: str,
    data_source: str,
    upload_example_table: bool,
) -> dict[str, Any]:
    # Convert all text to lowercase for easier matching
    texts_lower = [t.lower() for t in response_texts]
    total_count = len(texts_lower)

    # Identify whether each text contains reflection words for ratio computation
    has_words = [any(word in text for word in words) for text in texts_lower]

    has_word_ids = [ids[i] for i in range(len(ids)) if has_words[i] and is_correct[i]]
    has_word_correct_responses = [
        response_texts[i] for i in range(len(response_texts)) if has_words[i] and is_correct[i]
    ]
    has_word_correct_prompts = [
        prompt_texts[i] for i in range(len(prompt_texts)) if has_words[i] and is_correct[i]
    ]
    no_word_ids = [ids[i] for i in range(len(ids)) if (not has_words[i]) and is_correct[i]]
    no_word_correct_responses = [
        response_texts[i]
        for i in range(len(response_texts))
        if (not has_words[i]) and is_correct[i]
    ]
    no_word_correct_prompts = [
        prompt_texts[i] for i in range(len(prompt_texts)) if (not has_words[i]) and is_correct[i]
    ]

    example_dict = {}
    if upload_example_table:
        has_word_example_table = wandb.Table(columns=["step", "id", "prompt", "response"])
        if len(has_word_correct_responses) > 0 and len(has_word_correct_prompts) > 0:
            has_word_example_table.add_data(
                step, has_word_ids[0], has_word_correct_prompts[0], has_word_correct_responses[0]
            )

        no_word_example_table = wandb.Table(columns=["step", "id", "prompt", "response"])
        if len(no_word_correct_responses) > 0 and len(no_word_correct_prompts) > 0:
            no_word_example_table.add_data(
                step, no_word_ids[0], no_word_correct_prompts[0], no_word_correct_responses[0]
            )

        example_dict = {
            f"has_{metric_name}_correct_examples_{data_source}/step_{step}": has_word_example_table,
            f"no_{metric_name}_correct_examples_{data_source}/step_{step}": no_word_example_table,
        }

    # Count total, correct, incorrect, and reflection-included samples
    total_correct = sum(is_correct)
    total_incorrect = total_count - total_correct
    word_count = sum(has_words)

    # 1. Ratio of responses that contain at least one reflection word
    word_ratio = word_count / total_count if total_count else 0.0

    # 2. Among correct responses, ratio that contain reflection words
    if total_correct > 0:
        correct_with_word_count = sum(has_words[i] for i in range(total_count) if is_correct[i])
        word_ratio_in_correct_answers = correct_with_word_count / total_correct
    else:
        word_ratio_in_correct_answers = 0.0

    # 3. Among incorrect responses, ratio that contain reflection words
    if total_incorrect > 0:
        incorrect_with_word_count = sum(
            has_words[i] for i in range(total_count) if not is_correct[i]
        )
        word_ratio_in_incorrect_answers = incorrect_with_word_count / total_incorrect
    else:
        word_ratio_in_incorrect_answers = 0.0

    # 4. Among responses with reflection words, ratio that are correct
    if word_count > 0:
        correct_in_word_texts_count = sum(is_correct[i] for i in range(total_count) if has_words[i])
        correct_ratio_in_word_texts = correct_in_word_texts_count / word_count
    else:
        correct_ratio_in_word_texts = 0.0

    # 5. Among responses without reflection words, ratio that are correct
    no_word_count = total_count - word_count
    if no_word_count > 0:
        correct_in_no_word_texts_count = sum(
            is_correct[i] for i in range(total_count) if not has_words[i]
        )
        correct_ratio_in_no_word_texts = correct_in_no_word_texts_count / no_word_count
    else:
        correct_ratio_in_no_word_texts = 0.0

    # (A) Aggregate all computed statistics
    word_ratio_dict = {
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio": word_ratio,
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio_in_correct_answers": word_ratio_in_correct_answers,
        f"{metric_name}_ratios_{data_source}/{metric_name}_ratio_in_incorrect_answers": word_ratio_in_incorrect_answers,
        f"{metric_name}_ratios_{data_source}/correct_ratio_in_{metric_name}_texts": correct_ratio_in_word_texts,
        f"{metric_name}_ratios_{data_source}/correct_ratio_in_no_{metric_name}_texts": correct_ratio_in_no_word_texts,
    }
    # (B) Count total occurrences of each reflection word (accumulated across texts)
    word_frequency = {
        f"{metric_name}_words_{data_source}/{word}": sum(text.count(word) for text in texts_lower)
        for word in words
    }

    return_dict = {
        **word_ratio_dict,
        **word_frequency,
        **example_dict,
    }
    return return_dict


def _compute_tool_status(result_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    计算一个batch中工具使用的统计信息

    统计 intrinsic reward 相关的 N_A, N_B, N_C, N_D 数量：
    - N_A: Call Tool 并且 Correct
    - N_B: No Tool 并且 Wrong
    - N_C: Call Tool 并且 Wrong
    - N_D: No Tool 并且 Correct
    """
    # 统计各个类别的数量
    category_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for result_dict in result_dicts:
        category = result_dict.get("intrinsic_category", "")
        if category in category_counts:
            category_counts[category] += 1

    total_samples = len(result_dicts)

    # 保留 N_A, N_B, N_C, N_D 的统计
    N_A, N_B, N_C, N_D = (
        category_counts["A"],
        category_counts["B"],
        category_counts["C"],
        category_counts["D"],
    )

    # 从 result_dicts 中获取 TNS 和 TEQ 的平均值（避免重复计算）
    tns_values = [result_dict.get("TNS", 0.0) for result_dict in result_dicts]
    teq_values = [result_dict.get("TEQ", 0.0) for result_dict in result_dicts]
    tns_prime_values = [result_dict.get("TNS_prime", 0.0) for result_dict in result_dicts]
    p_calls = [result_dict.get("p_call", 0.0) for result_dict in result_dicts]
    p_nocalls = [result_dict.get("p_nocall", 0.0) for result_dict in result_dicts]

    avg_tns = sum(tns_values) / len(tns_values) if tns_values else 0.0
    avg_teq = sum(teq_values) / len(teq_values) if teq_values else 0.0
    avg_tns_prime = sum(tns_prime_values) / len(tns_prime_values) if tns_prime_values else 0.0
    avg_p_call = sum(p_calls) / len(p_calls) if p_calls else 0.0
    avg_p_nocall = sum(p_nocalls) / len(p_nocalls) if p_nocalls else 0.0

    return {
        "tool_intrinsic/N_A": N_A,
        "tool_intrinsic/N_B": N_B,
        "tool_intrinsic/N_C": N_C,
        "tool_intrinsic/N_D": N_D,
        "tool_intrinsic/total_samples": total_samples,
        "tool_intrinsic/TNS": avg_tns,
        "tool_intrinsic/TEQ": avg_teq,
        "tool_intrinsic/TNS_PRIME": avg_tns_prime,
        "tool_intrinsic/P_CALL": avg_p_call,
        "tool_intrinsic/P_NOCALL": avg_p_nocall,
        "tool_intrinsic/tool_usage_rate": (N_A + N_C) / total_samples if total_samples > 0 else 0.0,
        "tool_intrinsic/tool_success_rate": N_A / (N_A + N_C) if (N_A + N_C) > 0 else 0.0,
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str],
    sample_uids: list[str],
    infos_dict: dict[str, list[Any]],
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_uids: List of sample uids corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_uids = ["uid1", "uid1", "uid2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2uid2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        uid = sample_uids[sample_idx]
        var2vals = data_src2uid2var2vals[data_source][uid]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2uid2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, uid2var2vals in data_src2uid2var2vals.items():
        for uid, var2vals in uid2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                if isinstance(var_vals[0], dict):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                            data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                        )
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [
                                {"val": val, "pred": pred}
                                for val, pred in zip(var_vals, var2vals["pred"], strict=True)
                            ]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2uid2var2metric[data_source][uid][var_name] = metric

    # Aggregate metrics across uids
    data_src2var2metric2uid_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, uid2var2metric in data_src2uid2var2metric.items():
        for uid, var2metric in uid2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2uid_vals[data_source][var_name][metric_name].append(
                        metric_val
                    )

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2uid_vals in data_src2var2metric2uid_vals.items():
        for var_name, metric2uid_vals in var2metric2uid_vals.items():
            for metric_name, uid_vals in metric2uid_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(uid_vals)

    return data_src2var2metric2val
