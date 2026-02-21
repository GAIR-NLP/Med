import asyncio
import logging
import random
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import aiohttp
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleStepRemoteProxyManager:
    def __init__(self, rm_job, rm_num, rm_port, rm_fun):
        self.reward_server_job = rm_job
        self.reward_server_num = int(rm_num)
        self.reward_server_port = rm_port
        self.reward_server_function = rm_fun
        # Dict to track active connections per server
        self.active_connections = defaultdict(int)
        # Lock for thread-safe access to active_connections
        self.lock = threading.Lock()

        self.client_init()

    def client_init(self):
        def init_urls(job_ids_str, path):
            # --- 动态确定基础目录 (与原版相同) ---
            try:
                # 获取当前 Python 脚本的绝对路径
                # 注意：如果在交互式环境或某些打包应用中运行，__file__ 可能未定义
                current_file_path = Path(__file__).resolve()

                # 向上导航 4 级以获取 'verl' 的父目录
                project_root_dir = current_file_path.parents[3]

                # 构建 .reward_server 目录的路径
                base_dir = project_root_dir / ".reward_server"

                logger.info(f"Dynamically set base_dir to: {base_dir}")

            except (IndexError, NameError):  # 捕获 NameError 以防 __file__ 未定义
                logger.error(
                    "Could not determine base directory. "
                    "Is the file structure as expected "
                    "(verl/workers/reward_manager/remote_proxy.py)? "
                    "Falling back to '.reward_server'."
                )
                base_dir = Path(".reward_server")

            all_urls = []  # 用于存储所有找到的 URL

            # 去除后缀开头的斜杠，防止 URL 中出现 //
            clean_path = path.lstrip("/")

            # --- 分割 job_ids 字符串并处理 ---
            # 使用列表推导式分割字符串，并去除可能的空格，忽略空字符串
            job_id_list = [job_id.strip() for job_id in job_ids_str.split("|") if job_id.strip()]

            if not job_id_list:
                logger.warning("No valid job_ids provided.")
                return all_urls

            logger.info(f"Processing job_ids: {job_id_list}")

            # --- 遍历每一个 job_id ---
            for job_id in job_id_list:
                job_dir = base_dir / job_id
                logger.info(
                    f"Attempting to read server addresses for job '{job_id}' from directory: {job_dir.resolve()}..."
                )

                # 检查 Job 目录是否存在
                if not job_dir.is_dir():
                    logger.warning(f"Job directory not found for job_id '{job_id}': {job_dir}")
                    continue  # 如果目录不存在，跳过当前 job_id，继续下一个

                found_for_current_job = False
                # 遍历目录中的所有条目
                for item in job_dir.iterdir():
                    # 确保它是一个文件
                    if item.is_file():
                        ip_port = item.name  # 获取文件名 "IP:PORT"

                        # 检查文件名格式
                        if ":" not in ip_port or len(ip_port.split(":")) != 2:
                            logger.warning(
                                f"Skipping invalid filename in {job_dir}: {ip_port}, expected format IP:PORT"
                            )
                            continue

                        # 构建 URL
                        url = f"http://{ip_port}/{clean_path}"
                        all_urls.append(url)
                        logger.info(f"Added URL: {url} (from job_id: {job_id})")
                        found_for_current_job = True

                if not found_for_current_job:
                    logger.warning(
                        f"No server address files found in directory {job_dir} for job_id '{job_id}'."
                    )

            # --- 最终检查 ---
            if not all_urls:
                logger.warning(
                    f"No server address files found for any of the provided job_ids: {job_ids_str}"
                )

            return all_urls

        def verify_server(url):
            """Validate if server is ready by checking the root endpoint."""
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("message") == "Reward Judge Server":
                        return True
                    else:
                        print(f"Bad response from {url}")
                        return False
                else:
                    print(f"Status {response.status_code} from {url}")
                    return False
            except Exception as e:
                print(f"Error with {url}: {e}")
                return False

        # Initialize server URLs
        all_server_ips = init_urls(self.reward_server_job, self.reward_server_function)

        # Verify all servers
        verified_servers = []
        for server_ip in all_server_ips:
            root_server_ip = server_ip.split(self.reward_server_function)[0]
            is_verified = verify_server(root_server_ip)
            if is_verified:
                verified_servers.append(server_ip)

        if not verified_servers:
            raise RuntimeError("No reward servers could be verified")

        self.verified_servers = verified_servers

    def maintain_load_balance(self):
        """
        Select a random server from the verified servers list.
        This method returns a randomly chosen server URL.
        """
        if not self.verified_servers:
            raise RuntimeError("No verified servers available")

        # Select a random server
        # with self.lock:
        #     selected_server = random.choice(self.verified_servers)

        #     # Still track connections for monitoring purposes
        #     self.active_connections[selected_server] += 1
        selected_server = random.choice(self.verified_servers)
        return selected_server

    def release_server(self, server_url):
        """
        Release a server connection after use.
        """
        # with self.lock:
        #     if server_url in self.active_connections:
        #         self.active_connections[server_url] = max(0, self.active_connections[server_url] - 1)
        pass

    async def _send_request_with_retry(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
        max_retries: int = 3,
        timeout: int = 360,
    ) -> Optional[dict]:
        """
        Send a request to a server with retry logic.

        Args:
            session: aiohttp client session
            url: Server URL
            payload: Request payload
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds

        Returns:
            Response data or None if all retries failed
        """
        try:
            retries = 0
            while retries < max_retries:
                try:
                    async with session.post(url, json=payload, timeout=timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            print(f"Error response from {url}: {response.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Request to {url} failed: {e}")

                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(1)

            return None
        finally:
            # Always release the server when done
            self.release_server(url)

    async def _get_reward_async(self, payloads: list[dict[str, Any]]) -> list[Optional[dict]]:
        """
        Async implementation of get_reward.

        Args:
            payloads: List of payload dictionaries

        Returns:
            List of response data dictionaries
        """
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for payload in payloads:
                server_url = self.maintain_load_balance()
                task = asyncio.create_task(
                    self._send_request_with_retry(session, server_url, payload)
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        return results

    def get_reward(self, payloads: list[dict[str, Any]]) -> list[Optional[dict]]:
        """
        Send payloads to reward servers for evaluation.

        Args:
            payloads: List of payload dictionaries

        Returns:
            List of response data dictionaries
        """
        try:
            # Use asyncio.run instead of directly accessing the event loop
            return asyncio.run(self._get_reward_async(payloads))
        except Exception as e:
            print(f"Error in get_reward: {e}")
            return [None] * len(payloads)


if __name__ == "__main__":
    agent_manager = SingleStepRemoteProxyManager(
        rm_job="j-ldsch7okfx",
        rm_num=1,
        rm_port=8192,
        rm_fun="/judge",
    )
    print(agent_manager.verified_servers)

    reward_verifier_parm = {
        "verifier_style": "style",
        "image_grid_thw": [[1, 32, 64]],
        "det_verifier_normalized": True,
    }

    sample_payload = {
        "data_source": "object365_train",
        "query": None,
        "prompt": "system\nYou are a helpful assistant.\nuser\n\nIdentify all occurrences of the specified category in the image:\n- Car\n- Ambulance\n- Person\n- Trash bin Can\n\nBegin by clearly explaining your thought process enclosed within <think> and </think> tags. Afterward, present your final detection results enclosed within <answer> and </answer> tags.\nFor example:\n<think>\nYour detailed reasoning process here.\n</think>\n<answer>\n[{'bbox_2d': [x1,y1,x2,y2],'label': label_name}]\n</answer>\nassistant\n",
        "answer": "[{'bbox_2d': [447, 358, 508, 494], 'label': 'Person'}, {'bbox_2d': [0, 471, 111, 598], 'label': 'Car'}, {'bbox_2d': [108, 116, 874, 923], 'label': 'Ambulance'}, {'bbox_2d': [827, 524, 889, 687], 'label': 'Trash bin Can'}, {'bbox_2d': [916, 523, 999, 699], 'label': 'Trash bin Can'}]",
        "solution": "<answer>\n[{'bbox_2d': [447, 358, 508, 494], 'label': 'Person'}, {'bbox_2d': [0, 471, 111, 598], 'label': 'Car'}, {'bbox_2d': [108, 116, 874, 923], 'label': 'Ambulance'}, {'bbox_2d': [827, 524, 889, 687], 'label': 'Trash bin Can'}, {'bbox_2d': [916, 523, 999, 699], 'label': 'Trash bin Can'}]\n</answer>",
        "response": "<think>\nTo identify the specified categories, I'll analyze the image step-by-step:\n\n1. **Car**: This step involves detecting vehicles like cars. In the image, there is a vehicle in the background to the left which appears to be a parked car.\n\n2. **Ambulance**: This involves identifying vehicles that are specifically marked as ambulances. The vehicle in the foreground is yellow and green with \"AMBULANCE\" written on it, which is a clear indicator of an ambulance.\n\n3. **Person**: Observing the ambulance, there appears to be a figure inside, possibly a person. However, details are not clear due to the angle and distance.\n\n4. **Trash bin Can**: There is a yellow object in the background on the right side which looks like a trash bin due to its shape and location.\n\n</think>\n<answer>\n[{'bbox_2d': [0,313,45,386], 'label': 'Car'},\n {'bbox_2d': [16,317,116,396], 'label': 'Car'},\n {'bbox_2d': [0,326,32,403], 'label': 'Car'},\n {'bbox_2d': [955,346,1036,470], 'label': 'Trash bin Can'},\n {'bbox_2d': [59,348,89,414], 'label': 'Trash bin Can'}]\n</answer>",
        "reward_verifier": "detection",
        "reward_verifier_parm": '{"det_verifier_normalized": true, "det_reward_ratio": {"iou_max_label_first": 1.0, "iou_max_iou_first": 0.0, "iou_completeness": 0.30000001192092896, "map": 0.0, "map50": 0.0, "map75": 0.0}, "verifier_style": "rule", "is_training": true, "step": 1, "total_steps": 460, "image_grid_thw": [[1, 672, 1036]], "image_path": ["images/v2/patch16/objects365_v2_00902475.jpg"], "query": null}',
        "fmt_ratio": 0.10000000149011612,
        "len_ratio": 0.0,
    }

    rewards = agent_manager.get_reward([sample_payload])
    gather_rewards = rewards[0]["rewards"]
    score = float(gather_rewards["final_reward"])

    print("Reward: ", rewards, "gather_rewards: ", gather_rewards, "score: ", score)
