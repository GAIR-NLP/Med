from huggingface_hub import HfApi

api = HfApi()

# 创建仓库（如果已存在则跳过）
# api.create_repo(
#     repo_id="ManTle/vision_tool_use_dataset",
#     repo_type="dataset",
#     exist_ok=True
# )

# 上传整个文件夹
api.upload_folder(
    folder_path="/jfs-dialogue-mmos02-rs02/workspace/users/yema/code/verl_data/experiment_data",
    repo_id="ManTle/vision_tool_use_dataset",
    repo_type="dataset",
    path_in_repo="experiment_data",
)
