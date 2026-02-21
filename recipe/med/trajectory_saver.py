import json
import os
import time

import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils import hf_processor


def compress_sequence(seq):
    """压缩位置序列"""
    if not seq:
        return []

    compressed = []
    i = 0
    while i < len(seq):
        current_val = seq[i]

        # 检查连续递增
        j = i + 1
        while j < len(seq) and seq[j] == seq[j - 1] + 1:
            j += 1
        if j > i + 2:  # 至少3个连续才压缩
            compressed.append(f"{current_val}-{seq[j-1]}")
            i = j
            continue

        # 检查重复
        j = i + 1
        while j < len(seq) and seq[j] == current_val:
            j += 1
        if j > i + 2:  # 至少3个重复才压缩
            compressed.append(f"{current_val}x{j-i}")
            i = j
            continue

        # 单个值或短序列
        compressed.append(current_val)
        i += 1

    return compressed


def extract_and_compress_positions(sample, start_idx, end_idx):
    """提取并压缩position_ids"""
    position_ids = sample.batch["position_ids"]

    # 判断维度
    if position_ids.dim() == 1:
        # 1D position_ids: [seq_len]
        valid_positions = position_ids[start_idx:end_idx]
        return compress_sequence(valid_positions.tolist())

    elif position_ids.dim() == 2 and position_ids.shape[0] == 3:
        # 3D position_ids: [3, seq_len] - 只取第一维
        valid_positions = position_ids[0, start_idx:end_idx]  # 只要第0维
        return compress_sequence(valid_positions.tolist())

    else:
        # 其他情况，返回原始数据
        return position_ids[:, start_idx:end_idx].tolist()


def extract_and_save_messages(sample, dump_path, exp_name, step, save_images=False):
    """提取messages并保存图像文件"""
    uid = sample.non_tensor_batch["request_id"]
    messages = sample.non_tensor_batch["messages"]
    multi_modal_data = sample.non_tensor_batch.get("multi_modal_data", {})

    # 获取图像列表
    images = multi_modal_data.get("image", [])  # list of PIL.Image
    _ = multi_modal_data.get("video", [])  # 暂时不处理video

    # 构建目录结构: dump_path/exp_name/exp_name_step/
    trajectory_dir = os.path.join(dump_path, exp_name, f"{exp_name}_{step}") if dump_path else None
    images_dir = os.path.join(trajectory_dir, "images") if trajectory_dir else None
    if save_images and images_dir:
        os.makedirs(images_dir, exist_ok=True)

    processed_messages = []
    image_counter = 0

    for msg_idx, msg in enumerate(messages["messages"]):
        # 转换Message对象为dict
        msg_dict = msg.dump() if hasattr(msg, "dump") else dict(msg)

        processed_msg = {
            "role": msg_dict["role"],
            "content": [],
            "tool_calls": msg_dict.get("tool_calls", None),
        }

        # 处理content
        if isinstance(msg_dict["content"], list):
            for item in msg_dict["content"]:
                if item["type"] == "image":
                    if image_counter < len(images):
                        if save_images and images_dir:
                            # 保存PIL图像
                            image_filename = f"{uid}_msg{msg_idx}_img{image_counter}.png"
                            image_path = os.path.join(images_dir, image_filename)

                            # 直接保存PIL Image
                            pil_image = images[image_counter]
                            pil_image.save(image_path)

                            # 在content中记录相对路径
                            processed_msg["content"].append(
                                {
                                    "type": "image",
                                    "image_path": f"images/{image_filename}",
                                    "description": f"Image {image_counter + 1}",
                                }
                            )
                        else:
                            # 不保存图片，只记录占位符
                            processed_msg["content"].append(
                                {
                                    "type": "image_placeholder",
                                    "description": f"Image {image_counter + 1} (not saved)",
                                }
                            )
                        image_counter += 1
                    else:
                        processed_msg["content"].append(
                            {
                                "type": "image_placeholder",
                                "description": "Image not found",
                            }
                        )

                elif item["type"] == "text":
                    processed_msg["content"].append({"type": "text", "text": item["text"]})

        elif isinstance(msg_dict["content"], str):
            processed_msg["content"].append({"type": "text", "text": msg_dict["content"]})

        processed_messages.append(processed_msg)

    return processed_messages, trajectory_dir


def convert_to_json_serializable(obj):
    """递归转换对象为JSON可序列化格式"""
    if isinstance(obj, (torch.Tensor,)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "item"):  # numpy scalar types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        # 对于其他类型，尝试转换为string
        return str(obj)


def save_trajectories_jsonl(
    trajectories_list, samples_list, trajectory_dir, category, result_dicts_list=None
):
    """保存轨迹列表为JSONL文件"""
    jsonl_file = os.path.join(trajectory_dir, f"{category}_trajectories.jsonl")

    with open(jsonl_file, "w", encoding="utf-8") as f:
        for i, traj in enumerate(trajectories_list):
            # 添加元数据
            response_length = samples_list[i].batch["response_length"]
            tool_call_counts = samples_list[i].non_tensor_batch.get("tool_call_counts", 0)

            traj_with_meta = {
                "uid": str(samples_list[i].non_tensor_batch["request_id"]),
                "category": category,
                "response_length": (
                    int(response_length.item())
                    if torch.is_tensor(response_length)
                    else int(response_length)
                ),
                "tool_call_counts": int(tool_call_counts),
                **traj,  # 包含 full_valid_text, marked_text, position_ids, messages
            }

            # 添加所有包含"reward"的key
            if result_dicts_list is not None and i < len(result_dicts_list):
                for key, value in result_dicts_list[i].items():
                    if "reward" in key.lower():
                        traj_with_meta[key] = value

            # 添加对应的result_dict数据
            if result_dicts_list is not None and i < len(result_dicts_list):
                traj_with_meta["result_dict"] = result_dicts_list[i]

            # 移除trajectory_dir字段，避免重复
            traj_with_meta.pop("trajectory_dir", None)

            # 转换为JSON可序列化格式
            traj_serializable = convert_to_json_serializable(traj_with_meta)

            # 写入一行JSON
            f.write(json.dumps(traj_serializable, ensure_ascii=False) + "\n")

    return jsonl_file


def extract_text_data(sample, tokenizer, dump_path, exp_name, step, save_images=False):
    """Extract text data with colored segments based on response_mask"""
    input_ids = sample.batch["input_ids"]
    attention_mask = sample.batch["attention_mask"]
    prompt_length = sample.batch["prompts"].size(0)
    response_mask = torch.cat([torch.zeros(prompt_length), sample.batch["response_mask"]])

    # 找到有效内容区间(去padding)
    valid_indices = torch.where(attention_mask == 1)[0]
    start_idx = valid_indices[0].item()
    end_idx = valid_indices[-1].item() + 1

    # 提取有效的input_ids和对应的response_mask
    valid_input_ids = input_ids[start_idx:end_idx]
    valid_response_mask = response_mask[start_idx:end_idx]

    # 找到连续的0和1子序列
    segments = []
    current_value = None
    current_start = 0

    for i, mask_val in enumerate(valid_response_mask):
        if current_value is None:
            current_value = mask_val.item()
            current_start = i
        elif mask_val.item() != current_value:
            # 当前子序列结束，保存它
            segments.append(
                {
                    "mask_value": current_value,
                    "start_idx": current_start,
                    "end_idx": i,
                    "input_ids": valid_input_ids[current_start:i],
                }
            )
            current_value = mask_val.item()
            current_start = i

    # 处理最后一个子序列
    if current_value is not None:
        segments.append(
            {
                "mask_value": current_value,
                "start_idx": current_start,
                "end_idx": len(valid_response_mask),
                "input_ids": valid_input_ids[current_start:],
            }
        )

    # 对每个子序列decode并添加删除线标记

    marked_texts = []
    for segment in segments:
        segment_text = tokenizer.decode(segment["input_ids"])
        if segment["mask_value"] == 0:
            # 0子序列用特殊标记（被mask的部分）
            marked_texts.append(f"<MASK_START>{segment_text}<MASK_END>")
        else:
            # 1子序列正常显示
            marked_texts.append(segment_text)

    # 完整有效文本（去padding）
    full_valid_text = tokenizer.decode(valid_input_ids)

    # 带删除线标记的文本
    marked_text = "".join(marked_texts)

    # 提取并压缩position_ids
    compressed_position_ids = extract_and_compress_positions(sample, start_idx, end_idx)

    # 提取messages并保存图像
    messages_data, trajectory_dir = extract_and_save_messages(
        sample, dump_path, exp_name, step, save_images
    )

    return {
        "full_valid_text": full_valid_text,
        "marked_text": marked_text,
        "position_ids": compressed_position_ids,  # 压缩后的位置信息
        "messages": messages_data,  # 新增messages数据
        "trajectory_dir": trajectory_dir,  # 新增轨迹目录路径
    }


def convert_batch_to_trajectory(
    batch: DataProto,  # Using string for type hint as DataProto is not defined
    result_dicts: list[dict],
    tokenizer: PreTrainedTokenizer,
    dump_path: str,
    exp_name: str,
    step: int,
):
    """
    Processes a batch of data, separates samples by tool call usage,
    analyzes them, and prints statistics with timing for each step.
    """
    # Start a timer for the entire function
    total_start_time = time.time()

    # --- 1. Calculate Response Length for all samples ---
    # This is more efficient than doing it separately for each subgroup.
    time_len_calc_start = time.time()
    batched_response_length = {"response_length": torch.sum(batch.batch["response_mask"], dim=1)}

    batched_response_length = DataProto.from_dict(batched_response_length)
    batch.union(batched_response_length)

    print(f"Time to calculate all response lengths: {time.time() - time_len_calc_start:.4f}s")

    # --- 2. Separate samples with and without tool calls ---
    time_separation_start = time.time()
    has_tool_call_samples = [
        (idx, sample)
        for idx, sample in enumerate(batch)
        if sample.non_tensor_batch.get("tool_call_counts", 0) > 0
    ]
    no_tool_call_samples = [
        (idx, sample)
        for idx, sample in enumerate(batch)
        if sample.non_tensor_batch.get("tool_call_counts", 0) == 0
    ]
    print(f"Time to separate samples: {time.time() - time_separation_start:.4f}s")
    print("-" * 30)

    # 计算最大允许的response长度
    max_allowed_response_length = batch.batch["responses"].shape[-1]

    # --- 3. Process Samples WITHOUT Tool Calls ---
    sample_categories = {}

    if no_tool_call_samples:
        time_no_tool_call_start = time.time()

        # 1. 不带工具里最长的5个
        sample_categories["no_tool_longest"] = sorted(
            no_tool_call_samples,
            key=lambda s: s[1].batch["response_length"],
            reverse=True,
        )[:5]
        print("1. Top 5 no-tool-call samples with longest response:")
        for i, (_, sample) in enumerate(sample_categories["no_tool_longest"]):
            print(
                f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                f"Response Length = {sample.batch['response_length']}"
            )

        # 2. 不带工具里最长且未达到最大长度的5个
        not_max_no_tool_samples = [
            (idx, s)
            for idx, s in no_tool_call_samples
            if s.batch["response_length"] < max_allowed_response_length
        ]
        if not_max_no_tool_samples:
            sample_categories["no_tool_longest_not_max"] = sorted(
                not_max_no_tool_samples,
                key=lambda s: s[1].batch["response_length"],
                reverse=True,
            )[:5]
            print(
                f"\n2. Top 5 no-tool-call samples (longest, not reaching max length {max_allowed_response_length}):"
            )
            for i, (_, sample) in enumerate(sample_categories["no_tool_longest_not_max"]):
                print(
                    f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                    f"Response Length = {sample.batch['response_length']}"
                )
        else:
            sample_categories["no_tool_longest_not_max"] = []
            print(
                f"\n2. No no-tool-call samples found that haven't reached max length {max_allowed_response_length}"
            )

        # 3. 不带工具里最长且未达到最大长度且答对的5个
        correct_not_max_no_tool_samples = [
            (idx, s)
            for idx, s in no_tool_call_samples
            if s.batch["response_length"] < max_allowed_response_length
            and result_dicts[idx].get("accuracy_reward", 0) == 1.0
        ]
        if correct_not_max_no_tool_samples:
            sample_categories["no_tool_longest_not_max_correct"] = sorted(
                correct_not_max_no_tool_samples,
                key=lambda s: s[1].batch["response_length"],
                reverse=True,
            )[:5]
            print("\n3. Top 5 no-tool-call samples (longest, not max length, correct):")
            for i, (_, sample) in enumerate(sample_categories["no_tool_longest_not_max_correct"]):
                print(
                    f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                    f"Response Length = {sample.batch['response_length']}"
                )
        else:
            sample_categories["no_tool_longest_not_max_correct"] = []
            print("\n3. No correct no-tool-call samples found that haven't reached max length")

        print(f"Time for 'No Tool Call' processing: {time.time() - time_no_tool_call_start:.4f}s")
        print("-" * 30)

    # --- 4. Process Samples WITH Tool Calls ---
    if has_tool_call_samples:
        time_with_tool_call_start = time.time()
        print("Processing samples with tool calls...")

        # 4. 带工具里最长的5个
        sample_categories["with_tool_longest"] = sorted(
            has_tool_call_samples,
            key=lambda s: s[1].batch["response_length"],
            reverse=True,
        )[:5]
        print("\n4. Top 5 with-tool-call samples with longest response:")
        for i, (_, sample) in enumerate(sample_categories["with_tool_longest"]):
            print(
                f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                f"Response Length = {sample.batch['response_length']}"
            )

        # 5. 带工具里最长且未达到最大长度的5个
        not_max_with_tool_samples = [
            (idx, s)
            for idx, s in has_tool_call_samples
            if s.batch["response_length"] < max_allowed_response_length
        ]
        if not_max_with_tool_samples:
            sample_categories["with_tool_longest_not_max"] = sorted(
                not_max_with_tool_samples,
                key=lambda s: s[1].batch["response_length"],
                reverse=True,
            )[:5]
            print(
                f"\n5. Top 5 with-tool-call samples (longest, not reaching max length {max_allowed_response_length}):"
            )
            for i, (_, sample) in enumerate(sample_categories["with_tool_longest_not_max"]):
                print(
                    f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                    f"Response Length = {sample.batch['response_length']}"
                )
        else:
            sample_categories["with_tool_longest_not_max"] = []
            print(
                f"\n5. No with-tool-call samples found that haven't reached max length {max_allowed_response_length}"
            )

        # 6. 带工具且最长且未达到最大长度且答对的5个
        correct_not_max_with_tool_samples = [
            (idx, s)
            for idx, s in has_tool_call_samples
            if s.batch["response_length"] < max_allowed_response_length
            and result_dicts[idx].get("accuracy_reward", 0) == 1.0
        ]
        if correct_not_max_with_tool_samples:
            sample_categories["with_tool_longest_not_max_correct"] = sorted(
                correct_not_max_with_tool_samples,
                key=lambda s: s[1].batch["response_length"],
                reverse=True,
            )[:5]
            print("\n6. Top 5 with-tool-call samples (longest, not max length, correct):")
            for i, (_, sample) in enumerate(sample_categories["with_tool_longest_not_max_correct"]):
                print(
                    f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                    f"Response Length = {sample.batch['response_length']}"
                )
        else:
            sample_categories["with_tool_longest_not_max_correct"] = []
            print("\n6. No correct with-tool-call samples found that haven't reached max length")

        # 7. 调用工具最多的5个
        sample_categories["most_tool_calls"] = sorted(
            has_tool_call_samples,
            key=lambda s: s[1].non_tensor_batch["tool_call_counts"],
            reverse=True,
        )[:5]
        print("\n7. Top 5 samples with most tool calls:")
        for i, (_, sample) in enumerate(sample_categories["most_tool_calls"]):
            print(
                f"  Rank {i+1}: UID = {sample.non_tensor_batch['request_id']}, "
                f"Tool Calls = {sample.non_tensor_batch['tool_call_counts']}, "
                f"Response Length = {sample.batch['response_length']}"
            )

        print("-" * 30)
        print(
            f"Total time for 'With Tool Call' processing: {time.time() - time_with_tool_call_start:.4f}s"
        )
        print("-" * 30)

    # --- Generate trajectories for all categories ---
    print("Generating trajectories for all categories...")
    trajectories = {}
    for category, samples in sample_categories.items():
        if samples:  # 只为非空的类别生成轨迹
            trajectories[category] = [
                extract_text_data(sample, tokenizer, dump_path, exp_name, step, save_images=False)
                for _, sample in samples
            ]
            print(f"Generated {len(trajectories[category])} trajectories for {category}")
        else:
            trajectories[category] = []
            print(f"No samples for {category}")

    print(f"\nTotal function execution time: {time.time() - total_start_time:.4f}s")

    # --- Save trajectories as JSONL files ---
    print("Saving trajectories as JSONL files...")
    for category, category_trajectories in trajectories.items():
        if category_trajectories:  # 只保存非空的轨迹
            base_dir = category_trajectories[0]["trajectory_dir"]
            # 提取samples和对应的result_dicts
            samples = [sample for _, sample in sample_categories[category]]
            sample_result_dicts = [result_dicts[idx] for idx, _ in sample_categories[category]]

            save_trajectories_jsonl(
                category_trajectories,
                samples,
                base_dir,
                category,
                sample_result_dicts,
            )
            print(f"Saved {category}_trajectories.jsonl with {len(category_trajectories)} samples")
        else:
            print(f"No trajectories to save for {category}")

    print("\nAll trajectory files saved successfully!")


if __name__ == "__main__":
    import pickle

    with open("batch.pkl", "rb") as f:
        batch = pickle.load(f)
    with open("result_dicts.pkl", "rb") as f:
        result_dicts = pickle.load(f)
    processor = hf_processor("/verl_model/Qwen2.5-VL-7B-Instruct/", use_fast=True)
    assert processor is not None

    convert_batch_to_trajectory(
        batch, result_dicts, processor.tokenizer, "./trajectories", "test_exp", 1
    )
