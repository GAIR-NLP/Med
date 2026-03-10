# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.prompt_pool import *
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

IMAGE_MIN_TOKEN_NUM = 256
IMAGE_MAX_TOKEN_NUM = 16384


class EvalDataset(RLHFDataset):
    """
    Evaluation dataset that inherits from RLHFDataset.

    This class extends RLHFDataset with custom evaluation-specific processing
    in the __getitem__ method.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        answer = row_dict["reward_model"]["answer"]
        verifier = row_dict["reward_model"]["verifier"]
        need_tools_kwargs: bool = row_dict.get("extra_info", {}).get("need_tools_kwargs", False)
        agent_name: bool = row_dict.get("agent_name", "single_turn_agent")

        if agent_name == "tool_agent":
            # A hack to enable new image_crop_and_zoom_in tool
            row_dict["extra_info"]["tools_kwargs"]["image_crop_and_zoom_in_tool"] = (
                row_dict["extra_info"]["tools_kwargs"].pop("crop_and_zoom")
            )
            tool_names = list(row_dict["extra_info"]["tools_kwargs"].keys())
            tool_prompt = get_tool_calling_prompt(tool_names)
            row_dict[self.prompt_key][-1]["content"] += "\n\n" + tool_prompt

        think_prompt = get_thinking_prompt(answer, verifier)
        row_dict[self.prompt_key][-1]["content"] += "\n\n" + think_prompt

        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (
                process_image, process_video, random_square_block_mask)

            raw_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            multi_modal_data = {}

            images = None
            if (
                self.image_key in row_dict
                and row_dict.get(self.image_key, None) is not None
            ):
                images = [
                    process_image({
                        "image": image,
                        "min_pixels": IMAGE_MIN_TOKEN_NUM * self.image_patch_size ** 2,
                        "max_pixels": IMAGE_MAX_TOKEN_NUM * self.image_patch_size ** 2,
                    }) for image in row_dict.pop(self.image_key)
                ]

                if self.config.get("enable_multimodal_mask", False) and self.is_train:
                    images = [random_square_block_mask(image) for image in images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            if (
                self.video_key in row_dict
                and row_dict.get(self.video_key, None) is not None
            ):
                videos = [
                    process_video(video) for video in row_dict.pop(self.video_key)
                ]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images if images else None,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.tokenizer(
                raw_prompt, return_tensors="pt", add_special_tokens=False
            )
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.processor is not None
            and "Qwen2VLImageProcessor"
            in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = (
                    raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
                )
            elif self.truncation == "error":
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get(
            "interaction_kwargs", {}
        )
        need_tools_kwargs = row_dict.get("extra_info", {}).get(
            "need_tools_kwargs", self.need_tools_kwargs
        )
        if need_tools_kwargs and not tools_kwargs:
            raise ValueError(
                "tools_kwargs is empty for index {}, data source: {}",
                index,
                row_dict["data_source"],
            )

        # Process tools_kwargs to set raw_query from prompt content
        if tools_kwargs:
            for _, tool_config in tools_kwargs.items():
                if isinstance(tool_config, dict) and "create_kwargs" in tool_config:
                    create_kwargs = tool_config["create_kwargs"]
                    if isinstance(create_kwargs, dict) and "raw_query" in create_kwargs:
                        raw_query = ""
                        for content in messages[0]["content"]:
                            if content["type"] == "text":
                                raw_query += content["text"]
                        create_kwargs["raw_query"] = raw_query.strip()

        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
