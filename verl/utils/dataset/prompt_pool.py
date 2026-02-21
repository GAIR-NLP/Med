import random


def is_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def is_single_uppercase_letter(char_str: str):
    if len(char_str) != 1:
        return False
    return char_str.isalpha() and char_str.isupper()


mathverify_multi_choice_sentence = [
    "Answer the preceding multiple choice question. The last line of your response should follow "
    "this format: 'The final answer is: $\\boxed{LETTER}$' (without quotes), where LETTER is one of the option letters. "
    "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
    "information provided. Avoid thinking indefinitely, you MUST provide your best guess even if"
    "unsure. Think step by step logically, considering all relevant information before answering."
]

mathverify_letter_answer_sentence_zh = [
    "回答前面的选择题。你的回答最后一行应遵循此格式：'最终答案是: $\\boxed{LETTER}$'（不含引号），其中 LETTER 是一个选项字母。"
    "如果你不确定或问题太复杂，请基于所提供的信息做出合理猜测。请避免长时间思考，即使不确定，也必须给出你认为最好的猜测。"
    "回答前，请进行逻辑性的分步思考，并考虑所有相关信息。"
]


mathverify_multi_choice_direct_sentence = [
    "You cannot use any tools in this response. Please answer the preceding multiple choice question directly. "
    "Your response should follow this format: 'The final answer is: $\\boxed{LETTER}$' (without quotes), where LETTER is one of the option letters. "
    "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
    "information provided. Avoid thinking indefinitely, you MUST provide your best guess even if unsure."
]

mathverify_freeform_direct_sentence = [
    "You cannot use any tools in this response. Place your answer in \\boxed{}.",
    "You cannot use any tools in this response. Present the answer inside \\boxed{}.",
    "You cannot use any tools in this response. Your final answer should be placed in \\boxed{}.",
    "You cannot use any tools in this response. Present the result in \\boxed{}.",
    "You cannot use any tools in this response. Put the answer in \\boxed{}.",
    "You cannot use any tools in this response. Enclose your final answer in \\boxed{}.",
    "You cannot use any tools in this response. The final answer should be boxed in \\boxed{}.",
    "You cannot use any tools in this response. Put your solution inside \\boxed{}.",
    "You cannot use any tools in this response. Display the answer in \\boxed{}.",
    "You cannot use any tools in this response. Put the final answer within \\boxed{}.",
    "You cannot use any tools in this response. Finally, place the answer inside \\boxed{}.",
    "You cannot use any tools in this response. Put your final answer in \\boxed{}.",
    "You cannot use any tools in this response. Conclude with your final result inside \\boxed{}.",
    "You cannot use any tools in this response. Your answer should be enclosed within \\boxed{}.",
    "You cannot use any tools in this response. Wrap your answer in \\boxed{}.",
    "You cannot use any tools in this response. Enclose the final result inside \\boxed{}.",
    "You cannot use any tools in this response. Present the final answer in a \\boxed{} format.",
    "You cannot use any tools in this response. The final answer must be shown within \\boxed{}.",
    "You cannot use any tools in this response. Conclude by boxing the final result in \\boxed{}.",
    "You cannot use any tools in this response. Please display the final answer enclosed in \\boxed{}.",
]


# Tool Sentence (Use the crop_and_zoom tool when necessary)
crop_and_zoom_tool_sentence = [
    "For questions that depend on small or unclear visual details, consider using the crop_and_zoom tool for a closer look.",
    "If the answer lies in fine or indistinct visual details, try using the crop_and_zoom tool to get a better view.",
    "When dealing with questions requiring analysis of minute or blurry image elements, please utilize the crop_and_zoom tool.",
    "To better address queries about small or hard-to-see details in an image, employ the crop_and_zoom tool for magnification.",
    "Hint: For questions involving tiny or unclear visual specifics, the crop_and_zoom tool can help you inspect the area more closely.",
    "Use the crop_and_zoom tool when you need to magnify a specific part of an image to answer questions about its fine details.",
    "Consider activating the crop_and_zoom tool whenever a question's resolution depends on examining small-scale or fuzzy visual information.",
    "When visual details are crucial but indistinct, remember that the crop_and_zoom tool is available for a magnified inspection.",
    "For a more accurate response to questions about subtle visual details, make use of the crop_and_zoom tool to examine the image up close.",
    "Should you encounter a question that hinges on small or blurry visual cues, the crop_and_zoom tool is recommended for a detailed look.",
]

crop_and_zoom_tool_sentence_zh = [
    "对于依赖微小或不清晰视觉细节的问题，请考虑使用 crop_and_zoom 工具进行仔细查看。",
    "如果答案在于精细或模糊的视觉细节，请尝试使用 crop_and_zoom 工具以获得更清晰的视图。",
    "当处理需要分析微小或模糊图像元素的问题时，请使用 crop_and_zoom 工具。",
    "为了更好地回答关于图像中微小或难以看清细节的问题，请使用 crop_and_zoom 工具进行放大查看。",
    "提示：对于涉及微小或不清晰视觉细节的问题，crop_and_zoom 工具可以帮助您更仔细地检查相关区域。",
    "当您需要放大图像的特定部分以回答关于其精细细节的问题时，请使用 crop_and_zoom 工具。",
    "每当问题的解决依赖于检查微小或模糊的视觉信息时，请考虑启用 crop_and_zoom 工具。",
    "当视觉细节至关重要但又不清晰时，请记住可以使用 crop_and_zoom 工具进行放大检查。",
    "为了更准确地回答关于微妙视觉细节的问题，请使用 crop_and_zoom 工具近距离检查图像。",
    "如果您遇到的问题取决于微小或模糊的视觉线索，建议使用 crop_and_zoom 工具进行详细查看。",
]

image_crop_and_zoom_in_tool_sentence = [
    "For questions that depend on small or unclear visual details, consider using the image_crop_and_zoom_in_tool tool for a closer look.",
    "If the answer lies in fine or indistinct visual details, try using the image_crop_and_zoom_in_tool tool to get a better view.",
    "When dealing with questions requiring analysis of minute or blurry image elements, please utilize the image_crop_and_zoom_in_tool tool.",
    "To better address queries about small or hard-to-see details in an image, employ the image_crop_and_zoom_in_tool tool for magnification.",
    "Hint: For questions involving tiny or unclear visual specifics, the image_crop_and_zoom_in_tool tool can help you inspect the area more closely.",
    "Use the image_crop_and_zoom_in_tool tool when you need to magnify a specific part of an image to answer questions about its fine details.",
    "Consider activating the image_crop_and_zoom_in_tool tool whenever a question's resolution depends on examining small-scale or fuzzy visual information.",
    "When visual details are crucial but indistinct, remember that the image_crop_and_zoom_in_tool tool is available for a magnified inspection.",
    "For a more accurate response to questions about subtle visual details, make use of the image_crop_and_zoom_in_tool tool to examine the image up close.",
    "Should you encounter a question that hinges on small or blurry visual cues, the image_crop_and_zoom_in_tool tool is recommended for a detailed look.",
]

image_image_crop_and_zoom_in_tool_sentence_zh = [
    "对于依赖微小或不清晰视觉细节的问题，请考虑使用 image_crop_and_zoom_in_tool 工具进行仔细查看。",
    "如果答案在于精细或模糊的视觉细节，请尝试使用 image_crop_and_zoom_in_tool 工具以获得更清晰的视图。",
    "当处理需要分析微小或模糊图像元素的问题时，请使用 image_crop_and_zoom_in_tool 工具。",
    "为了更好地回答关于图像中微小或难以看清细节的问题，请使用 image_crop_and_zoom_in_tool 工具进行放大查看。",
    "提示：对于涉及微小或不清晰视觉细节的问题，image_crop_and_zoom_in_tool 工具可以帮助您更仔细地检查相关区域。",
    "当您需要放大图像的特定部分以回答关于其精细细节的问题时，请使用 image_crop_and_zoom_in_tool 工具。",
    "每当问题的解决依赖于检查微小或模糊的视觉信息时，请考虑启用 image_crop_and_zoom_in_tool 工具。",
    "当视觉细节至关重要但又不清晰时，请记住可以使用 image_crop_and_zoom_in_tool 工具进行放大检查。",
    "为了更准确地回答关于微妙视觉细节的问题，请使用 image_crop_and_zoom_in_tool 工具近距离检查图像。",
    "如果您遇到的问题取决于微小或模糊的视觉线索，建议使用 image_crop_and_zoom_in_tool 工具进行详细查看。",
]

crop_and_zoom_tool_sentence_zh = [
    "对于依赖微小或不清晰视觉细节的问题，请考虑使用 crop_and_zoom 工具进行仔细查看。",
    "如果答案在于精细或模糊的视觉细节，请尝试使用 crop_and_zoom 工具以获得更清晰的视图。",
    "当处理需要分析微小或模糊图像元素的问题时，请使用 crop_and_zoom 工具。",
    "为了更好地回答关于图像中微小或难以看清细节的问题，请使用 crop_and_zoom 工具进行放大查看。",
    "提示：对于涉及微小或不清晰视觉细节的问题，crop_and_zoom 工具可以帮助您更仔细地检查相关区域。",
    "当您需要放大图像的特定部分以回答关于其精细细节的问题时，请使用 crop_and_zoom 工具。",
    "每当问题的解决依赖于检查微小或模糊的视觉信息时，请考虑启用 crop_and_zoom 工具。",
    "当视觉细节至关重要但又不清晰时，请记住可以使用 crop_and_zoom 工具进行放大检查。",
    "为了更准确地回答关于微妙视觉细节的问题，请使用 crop_and_zoom 工具近距离检查图像。",
    "如果您遇到的问题取决于微小或模糊的视觉线索，建议使用 crop_and_zoom 工具进行详细查看。",
]


# First Sentence (Reasoning Step by Step)
mathverify_freeform_first_sentence = [
    "Please provide a step-by-step explanation of your reasoning.",
    "Break down the problem and solve it step by step.",
    "Guide me through your thought process step by step.",
    "Elaborate your reasoning step by step.",
    "Solve this problem by reasoning through each step.",
    "Provide your reasoning in a detailed, step-by-step manner.",
    "Explain each step of your thought process clearly.",
    "Work through the solution, showing each step.",
    "Start from the basics and explain each step of the process.",
    "Walk me through the reasoning process step by step.",
    "Please solve this by explaining each step of the way.",
    "Take me through your reasoning process, step by step.",
    "Explain the solution in a detailed, sequential manner.",
    "Describe each step in the solution process clearly.",
    "Provide a breakdown of your reasoning, step by step.",
    "Please outline the steps you took to solve this.",
    "Detail your thought process and explain each step.",
    "Walk me through the logic behind your solution step by step.",
    "Step through your reasoning process carefully.",
    "Break down the solution into manageable steps and explain each.",
]

# Second Sentence (Boxing the Final Answer)
mathverify_freeform_second_sentence = [
    "Finally, place your answer in \\boxed{}. ",
    "Conclude your reasoning and present the answer inside \\boxed{}.",
    "Your final answer should be placed in \\boxed{}.",
    "Present the result in \\boxed{}.",
    "Wrap up your reasoning and put the solution in \\boxed{}.",
    "Enclose your final answer in \\boxed{}.",
    "The final answer should be boxed in \\boxed{}.",
    "Put your solution inside \\boxed{}.",
    "Display the answer in \\boxed{}.",
    "Put the final answer within \\boxed{}.",
    "Finally, place the answer inside \\boxed{}.",
    "Put your final answer in \\boxed{}.",
    "Conclude with your final result inside \\boxed{}.",
    "Your answer should be enclosed within \\boxed{}.",
    "Wrap your solution in \\boxed{}.",
    "Enclose the final result inside \\boxed{}.",
    "Present the final answer in a \\boxed{} format.",
    "The final answer must be shown within \\boxed{}.",
    "Conclude by boxing the final result in \\boxed{}.",
    "Please display the final answer enclosed in \\boxed{}.",
]

mathverify_closeform_first_sentence = [
    "Please solve the problem step by step, providing clear explanations. The reasoning should be enclosed within <think> </think> tags, while the final answer is to be placed within <answer> </answer> tags.",
    "Solve the problem by breaking it down step by step. The reasoning process should be enclosed in <think> </think> tags, and the final answer should be inside <answer> </answer> tags.",
    "Begin by solving the problem step by step, with detailed reasoning enclosed in <think> </think> tags. The final answer must be placed within <answer> </answer> tags.",
    "Please solve the problem sequentially, explaining each step clearly. Enclose your reasoning within <think> </think> tags, and place the final answer inside <answer> </answer> tags.",
    "Solve the problem in steps, providing reasoning within <think> </think> tags, and place the final answer within <answer> </answer> tags.",
    "Proceed to solve the problem step by step, making sure that the reasoning is enclosed in <think> </think> tags. The final answer should be in <answer> </answer> tags.",
    "Break down the problem and solve it step by step. The reasoning should be within <think> </think> tags, and the answer must be enclosed in <answer> </answer> tags.",
    "Work through the problem step by step, enclosing your reasoning in <think> </think> tags, and placing the final answer in <answer> </answer> tags.",
    "Step through the problem, ensuring that your reasoning is contained in <think> </think> tags, with the answer placed inside <answer> </answer> tags.",
    "Please solve the problem by explaining each step in detail. Your reasoning should be enclosed in <think> </think> tags, and the answer should be inside <answer> </answer> tags.",
]

mathverify_closeform_second_sentence = [
    "Moreover, the final answer should be inside \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "In addition, format the answer inside \\boxed{}. For instance: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Also, the final answer should be enclosed within \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Additionally, ensure the answer is in \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Furthermore, the final answer needs to be formatted inside \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Also, the answer itself must be enclosed in \\boxed{}. For instance: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Ensure the answer is formatted inside \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Additionally, the final answer must be placed inside \\boxed{}. For instance: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Make sure the answer is enclosed in \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
    "Lastly, format the answer within \\boxed{}. For example: <think>\nreasoning process here\n</think>\n<answer>\n\\boxed{answer here}\n</answer>.",
]

sft2rl_closeform_first_sentence = [
    "Please solve the problem step by step, providing clear explanations. Make sure your reasoning refers to the image and that each step aligns with the visual content. Enclose the reasoning within <think> </think> tags.",
    "Solve the problem by breaking it down step by step, grounding your reasoning in the visual information from the image. Each step must be consistent with the image and enclosed in <think> </think> tags.",
    "Begin by solving the problem step by step, using details from the image when appropriate. At every step, verify that your reasoning does not contradict the image. Provide your reasoning within <think> </think> tags.",
    "Please solve the problem sequentially, explaining each step clearly. Ensure your reasoning is grounded in the image and verify that it is consistent with what the image shows. Enclose it within <think> </think> tags.",
    "Solve the problem in steps, using the image to support your reasoning. Each intermediate step must be checked for consistency with the image. Enclose your reasoning within <think> </think> tags.",
    "Proceed to solve the problem step by step, making sure that your reasoning refers to the image content. For every step, confirm it aligns with the image, and enclose it in <think> </think> tags.",
    "Break down the problem and solve it step by step, grounding each part of your reasoning in the visual input. Verify at each point that the reasoning matches the image. The reasoning should be within <think> </think> tags.",
    "Work through the problem step by step, explicitly referring to the image as needed. For every reasoning step, ensure it does not contradict the visual information. Enclose your reasoning in <think> </think> tags.",
    "Step through the problem, ensuring that your reasoning is supported by the image and that each step remains consistent with what the image shows. Enclose it in <think> </think> tags.",
    "Please solve the problem by explaining each step in detail. Make sure to reason with the help of the image, and at every stage confirm that your reasoning agrees with the image. Enclose your reasoning in <think> </think> tags.",
]

sft2rl_closeform_reasonauto_sentence = [
    "In cases where reasoning is unnecessary because the answer is clear, keep the reasoning section blank using <think>\n</think>.",
    "If the problem is straightforward and doesn’t require reasoning, leave the section empty with <think>\n</think>.",
    "When no reasoning is necessary due to simplicity, keep the section empty using <think>\n</think>.",
    "If no detailed reasoning is needed, simply leave the section blank as <think>\n</think>.",
    "For simple problems requiring no explanation, leave the reasoning block empty with <think>\n</think>.",
    "If the problem is easy and requires no thought process, omit the content by using <think>\n</think>.",
    "When no reasoning is warranted due to the simplicity of the problem, use <think>\n</think> to indicate an empty reasoning section.",
    "For problems that can be answered directly without reasoning, represent the empty reasoning section with <think>\n</think>.",
    "When the answer is obvious and does not involve any reasoning steps, indicate the absence of reasoning with <think>\n</think>.",
    "If the question is simple and does not call for an explanation, leave the reasoning part empty as <think>\n</think>.",
]

sft2rl_mathverify_VRM_closeform_second_sentence = [
    "Moreover, the final answer should be inside \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "In addition, format the answer inside \\boxed{}. For instance:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Also, the final answer should be enclosed within \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Additionally, ensure the answer is in \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Furthermore, the final answer needs to be formatted inside \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Also, the answer itself must be enclosed in \\boxed{}. For instance:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Ensure the answer is formatted inside \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Additionally, the final answer must be placed inside \\boxed{}. For instance:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Make sure the answer is enclosed in \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
    "Lastly, format the answer within \\boxed{}. For example:\n<think>\nreasoning process here\n</think>\n\\boxed{answer here}.",
]

sft2rl_editdistance_closeform_second_sentence = [
    "Moreover, the final answer should be enclosed with <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "In addition, make sure to wrap the final answer within <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Also, the final answer must appear inside <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Additionally, format the answer by placing it between <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Furthermore, your answer needs to be wrapped in <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Also, ensure that the final answer goes within <answer> and </answer> tags. For instance:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Make sure to enclose your answer using <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Additionally, the answer should be clearly marked with <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Remember to insert the final answer inside <answer> and </answer> tags. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
    "Lastly, place your answer within <answer> and </answer> tags to indicate the final result. For example:\n<think>\nreasoning process here\n</think>\n<answer>\nanswer here\n</answer>.",
]

sft2rl_detection_closeform_first_sentence = [
    "Output the answer and its bounding box.",
    "Return the answer along with its bounding box.",
    "Provide the answer and its corresponding bounding box.",
    "Report the answer and specify its bounding box.",
    "Identify the answer and include its bounding box.",
    "Give the answer together with its bounding box.",
    "Submit the answer and annotate its bounding box.",
    "Present the answer and mark its bounding box.",
    "State the answer and indicate its bounding box.",
    "Extract the answer and show its bounding box.",
]

sft2rl_detection_closeform_second_sentence = [
    "Display the detection results clearly enclosed within <answer> and </answer> tags to emphasize the final output.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Afterward, give the final detection results wrapped in <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Lastly, present the detection results within <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Subsequently, enclose the final detection output in <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Provide the detection output formatted within <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "After that, wrap the detection result with <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "In the end, present the detection output enclosed by <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Additionally, format your detection output using <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Next, include the final detection results inside <answer> and </answer> tags.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
    "Finally, deliver your detection findings clearly by enclosing them within <answer> and </answer> tags, ensuring the results are easy to identify.\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>.",
]

sft2rl_closeform_first_sentence_zh = [
    "请逐步解决这个问题，并清晰解释每一步。推理过程中请结合图像信息，并确保每一步都与图像内容一致，不可出现矛盾。推理应放在 <think> </think> 标签中。",
    "请一步一步地解决问题，推理时注意结合图像内容，并确认每一步推理与图像不冲突。将推理过程用 <think> </think> 标签包裹。",
    "请按照步骤解决问题，推理过程应参考图像信息，并确保中间步骤与图像相符，不可自相矛盾。推理写在 <think> </think> 标签内。",
    "请依次解决问题，清楚地解释每个步骤。每一步推理都必须结合图像内容并保持一致，放入 <think> </think> 标签中。",
    "请分步骤解答该问题，推理应基于图像内容，且每步逻辑不能与图像信息矛盾。写在 <think> </think> 标签中。",
    "请一步一步地完成题目，推理时应参考图像信息，并逐步验证推理是否合理，确保与图像一致。将推理包含在 <think> </think> 标签中。",
    "请将问题分解后逐步解决，推理时请结合图像内容，并在每一步中检查推理是否与图像信息相符。将推理写在 <think> </think> 标签内。",
    "请按照步骤解答问题，推理时需依赖图像，并确保每一步都不违反图像所提供的信息。用 <think> </think> 标签括起。",
    "请逐步进行解题，确保推理过程结合图像内容，并在每个推理步骤中验证其与图像的一致性。写在 <think> </think> 标签中。",
    "请详细解释每一步解题过程，并在推理中参考图像信息。注意每一步都必须与图像信息一致，推理部分用 <think> </think> 标签包含。",
]

sft2rl_closeform_reasonauto_sentence_zh = [
    "当答案明确不需要推理时，保持推理部分为空，使用 <think>\n</think>。",
    "如果问题简单且无需推理，请用 <think>\n</think> 留空该部分。",
    "当不需要推理时，保持该部分为空，用 <think>\n</think> 表示。",
    "如果无需详细推理，只需将该部分留空，使用 <think>\n</think>。",
    "对于无需解释的简单问题，用 <think>\n</think> 留空推理块。",
    "当问题简单无需思考过程时，用 <think>\n</think> 省略内容。",
    "若因问题简单不需推理，可用 <think>\n</think> 表示推理部分为空。",
    "对于可直接回答且无需推理的问题，用 <think>\n</think> 表示空的推理部分。",
    "当答案显而易见且不涉及推理步骤时，用 <think>\n</think> 表示没有推理。",
    "如果问题简单不需解释，请将推理部分留空为 <think>\n</think>。",
]

sft2rl_mathverify_VRM_closeform_second_sentence_zh = [
    "此外，最终答案应写在 \\boxed{} 中。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "另外，请将最终答案格式变为 \\boxed{} 的形式。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "同时，最终答案必须包含在 \\boxed{} 中。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "还请将答案写入 \\boxed{} 中。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "请将最终答案格式为 \\boxed{} 的形式。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "请将答案放入 \\boxed{} 中，例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "请确保答案写在 \\boxed{} 中。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "最终答案必须用 \\boxed{} 包裹。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "请将答案放在 \\boxed{} 中。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
    "最后，请用 \\boxed{} 格式书写答案。例如：\n<think>\\n推理过程\\n</think>\\n\\boxed{答案}。",
]

sft2rl_editdistance_closeform_second_sentence_zh = [
    "此外，最终答案应写在 <answer> 和 </answer> 标签之间。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "另外，请将最终答案放入 <answer> 和 </answer> 标签中。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "同时，最终答案必须使用 <answer> 和 </answer> 标签进行标注。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "还请将答案内容包裹在 <answer> 和 </answer> 标签内。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "请将最终答案格式设为 <answer>标签</answer> 包含的形式。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "请将答案插入到 <answer> 与 </answer> 标签之中。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "请确保你的答案写在 <answer> 和 </answer> 标签之间。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "最终答案需要用 <answer> 和 </answer> 标签明确标出。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "请将答案写在 <answer> 和 </answer> 所标记的范围内。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
    "最后，请使用 <answer> 和 </answer> 标签来呈现你的答案。例如：\n<think>\\n推理过程\\n</think>\\n<answer>\\n答案\\n</answer>。",
]

sft2rl_detection_closeform_first_sentence_zh = [
    "输出答案及其边界框。",
    "返回答案及其对应的边界框。",
    "提供答案以及对应的边界框。",
    "报告答案并标明其边界框。",
    "识别出答案并包含其边界框。",
    "给出答案以及其边界框。",
    "提交答案并标注其边界框。",
    "展示答案并标记其边界框。",
    "说明答案并指出其边界框。",
    "抽取答案并展示其边界框。",
]

sft2rl_detection_closeform_second_sentence_zh = [
    "将检测结果清晰地包裹在 <answer> 和 </answer> 标签中，以突出最终输出。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "随后，将最终检测结果包裹在 <answer> 和 </answer> 标签中。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "最后，请将检测结果展示在 <answer> 和 </answer> 标签中。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "之后，将最终检测输出包裹在 <answer> 和 </answer> 标签中。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "请将检测输出以 <answer> 和 </answer> 标签的格式提供。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "接着，用 <answer> 和 </answer> 标签包裹检测结果。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "最后，请将检测输出用 <answer> 和 </answer> 标签包含。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "此外，请使用 <answer> 和 </answer> 标签格式化检测输出。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "接下来，请将最终检测结果写在 <answer> 和 </answer> 标签中。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
    "最后，请将检测结果清晰地包裹在 <answer> 和 </answer> 标签中，以便于识别。\n例如：\n<think>\n此处为推理过程\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>。",
]


def get_thinking_prompt(answer: str, verifier: str):
    if verifier == "mathverify":
        if is_single_uppercase_letter(answer):
            return random.choice(mathverify_multi_choice_sentence)
        else:
            first_sentence = random.choice(mathverify_freeform_first_sentence)
            second_sentence = random.choice(mathverify_freeform_second_sentence)
            return "\n".join([first_sentence, second_sentence])
    return ""


def get_direct_prompt(answer: str):
    if is_single_uppercase_letter(answer):
        return random.choice(mathverify_multi_choice_direct_sentence)
    else:
        return random.choice(mathverify_freeform_direct_sentence)


def get_tool_calling_prompt(tool_name_list: list[str]) -> str:
    available_tool_prompts = []
    for tool_name in tool_name_list:
        if tool_name == "image_crop_and_zoom_in_tool":
            available_tool_prompts.append(random.choice(image_crop_and_zoom_in_tool_sentence))
    return "\n\n".join(available_tool_prompts)
