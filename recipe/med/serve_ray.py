# main_ray.py - Final, robust version for Ray Serve

import base64
import io
import json
import logging
import os
from logging.handlers import RotatingFileHandler

# Ray and Ray Serve imports
import ray

# FastAPI and related imports
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ray import serve

# 假设您的函数在这个路径
from recipe.o3.vision_tool_fn import image_crop_and_zoom_in_tool, smart_crop_and_zoom

LOG_DIR = "/verl_vision/logs"  # <-- 指定一个你希望存放日志的目录
LOG_FILE = os.path.join(LOG_DIR, "vision_tool_service.log")

os.makedirs(LOG_DIR, exist_ok=True)

# 配置logger
logger = logging.getLogger("vision_tool")
logger.setLevel(logging.INFO)

# 创建一个文件处理器 (FileHandler)，设置日志滚动
# 这里设置为每个日志文件最大10MB，保留5个备份
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(process)d - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# 将文件处理器添加到logger
logger.addHandler(file_handler)

# ==============================================================================
# 1. 在顶层创建FastAPI应用实例
#    这使得Web路由和业务逻辑分离得更清晰
# ==============================================================================
app = FastAPI()


@app.get("/routes")
def get_all_registered_routes(request: Request):
    """
    一个调试端点，用来返回当前FastAPI应用中所有已注册的路由。
    """
    url_list = [
        {"path": route.path, "name": route.name, "methods": list(route.methods)}
        for route in request.app.routes
    ]
    return url_list


# ==============================================================================
# 2. 将FastAPI的路由直接绑定到顶层的 `app` 实例上
# ==============================================================================
@app.post("/crop_and_zoom")
async def crop_and_zoom_endpoint(
    image_file: UploadFile = File(...),
    crop_box: str = Form(...),
    zoom_factor: float | None = Form(None),
    output_resolution_limit: int | None = Form(256),
):
    try:
        contents = await image_file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        crop_box_tuple = tuple(json.loads(crop_box))

        result_dict = smart_crop_and_zoom(
            img=input_image,
            crop_box=crop_box_tuple,
            zoom_factor=zoom_factor,
            output_resolution_limit=output_resolution_limit,
        )
        processed_image = result_dict["image"]

        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(
            content={
                "processed_image_b64": img_b64,
                "message": result_dict["message"],
                "status": result_dict["status"],
                "tool_reward": result_dict["tool_reward"],
                "input_resolution": input_image.size,
                "output_resolution": processed_image.size,
                "ray_node_id": ray.get_runtime_context().get_node_id(),
                "ray_worker_id": ray.get_runtime_context().get_worker_id(),
            }
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for crop_box.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.post("/image_crop_and_zoom_in_tool")
async def image_crop_and_zoom_in_tool_endpoint(
    image_file: UploadFile = File(...),
    crop_box: str = Form(...),
    label: str = Form("cropped_region"),
):
    try:
        contents = await image_file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        bbox_2d_tuple = tuple(json.loads(crop_box))

        result_dict = image_crop_and_zoom_in_tool(
            img=input_image,
            bbox_2d=bbox_2d_tuple,
            label=label,
        )
        processed_image = result_dict["image"]

        if processed_image is not None and isinstance(processed_image, Image.Image):
            buffered = io.BytesIO()
            processed_image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            img_b64 = None

        return JSONResponse(
            content={
                "processed_image_b64": img_b64,
                "message": result_dict["message"],
                "status": result_dict["status"],
                "tool_reward": result_dict["tool_reward"],
                "label": label,
            }
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for bbox_2d.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# ==============================================================================
# 3. 定义一个部署类，但这次使用 @serve.ingress 装饰器
#    这个类现在可以专注于持有模型、状态等，而不是处理Web请求
# ==============================================================================
@serve.deployment(
    # Default number of replicas, will be overridden by the YAML config during deployment.
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)  # <--- 这是关键！告诉Serve将请求直接交给FastAPI的`app`处理
class VisionToolServer:
    # 构造函数现在可以用来加载模型等
    # 在这个例子里，我们不需要做任何事，所以可以留空或省略
    def __init__(self):
        logger.info("VisionToolServer replica initialized.")
        # 注意：这里不再有 self.app = FastAPI()

    # 关键：我们完全删除了 __call__ 方法！
    # 因为 @serve.ingress 已经接管了所有HTTP流量的处理


# ==============================================================================
# 4. 绑定部署，创建Serve应用入口
# ==============================================================================
vision_app = VisionToolServer.bind()
