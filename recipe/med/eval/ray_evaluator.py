import uuid
import os
import ray
import time
from collections import defaultdict
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager


@ray.remote(num_gpus=1)
class VerificationLLMActor:
    """Ray Actor for running LLM verification with vLLM."""
    
    def __init__(self, config: dict):
        from vllm import LLM, SamplingParams
        
        self.config = config
        
        # Initialize vLLM LLM class
        self.llm = LLM(
            model=config["model"],
            tensor_parallel_size=config.get("tensor_parallel_size", 1),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.8),
            max_model_len=config.get("max_model_len", 4096),
            dtype=config.get("dtype", "bfloat16"),
            trust_remote_code=config.get("trust_remote_code", True),
            enforce_eager=config.get("enforce_eager", False),
        )
        
        # Set up sampling parameters for verification
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic judgment
            top_p=1.0,
            max_tokens=config.get("max_tokens", 128),
        )
        
        print(f"✅ VerificationLLMActor (vLLM) initialized with model: {config['model']}")
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        responses = []
        for output in outputs:
            content = output.outputs[0].text.strip()
            responses.append(content)
        
        return responses
    
    def generate_single(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'sampling_params'):
            del self.sampling_params
        
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print("  VerificationLLMActor (vLLM) cleaned up")


@ray.remote(num_gpus=1)
class VerificationLLMActorSGLang:
    """Ray Actor for running LLM verification with SGLang."""
    
    def __init__(self, config: dict):
        import sglang as sgl
        
        self.config = config
        
        # Initialize SGLang engine
        self.llm = sgl.Engine(
            model_path=config["model"],
            tp_size=config.get("tensor_parallel_size", 1),
            mem_fraction_static=config.get("gpu_memory_utilization", 0.8),
            context_length=config.get("max_model_len", 25600),
            dtype=config.get("dtype", "bfloat16"),
            trust_remote_code=config.get("trust_remote_code", True),
        )
        
        # Store sampling parameters for verification
        self.sampling_params = {
            "temperature": 0.0,  # Deterministic judgment
            "top_p": 1.0,
            "max_new_tokens": config.get("max_tokens", 128),
        }
        
        print(f"✅ VerificationLLMActorSGLang (SGLang) initialized with model: {config['model']}")
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generate responses for a batch of prompts."""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output["text"].strip() for output in outputs]
    
    def generate_single(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        output = self.llm.generate([prompt], self.sampling_params)
        return output[0]["text"].strip()
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            self.llm.shutdown()
            del self.llm
        
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print("  VerificationLLMActorSGLang (SGLang) cleaned up")


class RayEvaluator(RayPPOTrainer):
    """Distributed evaluator using Ray for scalable evaluation.

    This evaluator inherits from RayPPOTrainer to reuse all the validation logic
    while providing a clean interface specifically for evaluation tasks.
    """

    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
        val_dataset: Dataset,
        processor: ProcessorMixin,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn=None,
    ):
        """Initialize distributed evaluator with Ray backend.

        Args:
            config: Configuration object containing evaluation parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            val_dataset: Validation dataset for evaluation.
            processor: Data processor for multimodal data.
            role_worker_mapping: Mapping from roles to worker classes.
            resource_pool_manager: Manager for Ray resource pools.
            ray_worker_group_cls: Class for Ray worker groups.
            reward_fn: Function for computing rewards during evaluation.
        """
        # Store essential components for evaluation
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.val_reward_fn = (
            reward_fn  # Store as val_reward_fn for compatibility with _validate
        )

        # Ray components
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = config.trainer.get("device", "cuda")

        # Create dataloader
        self._create_dataloader(None, val_dataset, None, None)
        
        print(f"=====DEBUG=====: self.config: {self.config}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Create only validation dataloader for evaluation."""
        if val_dataset is None:
            from verl.trainer.main_ppo import create_rl_dataset

            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
            )

        self.val_dataset = val_dataset

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import \
                collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data.get("dataloader_num_workers", 0)

        val_batch_size = self.config.data.get("val_batch_size", None)
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", False),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of validation dataloader: {len(self.val_dataloader)}")

    def init_workers(self):
        """Initialize only the necessary workers for evaluation (actor_rollout only)."""
        # Create resource pools
        self.resource_pool_manager.create_resource_pool()

        # Setup async rollout if needed
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.get("mode", "sync") == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(config=self.config)

        print("✅ Evaluation workers initialized successfully")

    def _validate(self):
        """Override parent _validate method to add pass@k calculation."""
        print(f"DEBUG: val_dataset length = {len(self.val_dataset)}")
        print(f"DEBUG: val_batch_size = {self.config.data.get('val_batch_size', None)}")
        print(f"DEBUG: actual batch_size used = {self.val_dataloader.batch_size}")
        print(f"DEBUG: dataloader length = {len(self.val_dataloader)}")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        # sample_scores = []
        sample_turns = []
        sample_result_dicts = []

        # Get k value for pass@k calculation
        k = self.config.actor_rollout_ref.rollout.val_kwargs.n
        print(f"Running evaluation with pass@{k}")

        start_time = time.time()
        test_batch_list = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
            )

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=k, interleave=True)

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, size_divisor
            )

            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                    test_gen_batch_padded
                )
            else:
                test_output_gen_batch_padded = (
                    self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            test_batch.meta_info["global_steps"] = self.global_steps
            test_batch.meta_info["total_steps"] = self.total_training_steps

            test_batch_list.append(test_batch)

        test_batch = DataProto.concat(test_batch_list)

        # evaluate using reward_function
        if self.val_reward_fn is None:
            raise ValueError("val_reward_fn must be provided for validation.")

        result = self.val_reward_fn(test_batch, return_dict=True)
        reward_tensor = result["reward_tensor"]

        # Get result dicts before processing
        result_dicts = result["reward_extra_info"]["result_dicts"]

        # LLM verification step (if enabled)
        if self._should_enable_llm_verification(result_dicts):
            print("Starting LLM verification...")
            result_dicts = self._llm_verify_accuracy(test_batch, result_dicts)
            print("LLM verification completed")

        # Add pass@k tags to result_dicts
        result_dicts_with_passk = self._add_passk_tags(result_dicts, k)
        sample_result_dicts.extend(result_dicts_with_passk)

        scores = reward_tensor.sum(-1).cpu().tolist()
        # sample_scores.extend(scores)
        reward_extra_infos_dict["reward"].extend(scores)
        print(
            f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}"
        )

        if "reward_extra_info" in result:
            for key, lst in result["reward_extra_info"].items():
                reward_extra_infos_dict[key].extend(lst)
                print(
                    f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}"
                )

        # collect num_turns of each prompt
        if "__num_turns__" in test_batch.non_tensor_batch:
            sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

        data_source_lst.append(
            test_batch.non_tensor_batch.get(
                "data_source", ["unknown"] * reward_tensor.shape[0]
            )
        )

        end_time = time.time()
        print(
            f"=======Benchmark {len(self.val_dataloader)} iter in test, takes: {round(end_time - start_time, 2)} seconds======="
        )

        # Process metrics with pass@k information
        metric_dict = self._compute_metrics_with_passk(sample_result_dicts, k)

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Store test_batch and sample_result_dicts for trajectory saving
        self._last_test_batch = test_batch
        self._last_sample_result_dicts = sample_result_dicts

        return metric_dict

    def _add_passk_tags(self, result_dicts, k):
        """Add pass@k tags to each sample in result_dicts."""
        result_dicts_with_passk = []

        for i, result_dict in enumerate(result_dicts):
            # Calculate which original sample this result belongs to
            original_sample_idx = i // k
            sample_group_start = original_sample_idx * k
            sample_group_end = sample_group_start + k

            # Get all results for this original sample
            sample_group_results = result_dicts[sample_group_start:sample_group_end]

            # Check if any result in this group is correct (pass@k logic)
            pass_at_k = any(
                group_result.get("accuracy_reward", 0) > 0
                for group_result in sample_group_results
            )

            # Create new result dict with pass@k tag
            new_result_dict = result_dict.copy()
            new_result_dict["pass_at_k"] = pass_at_k
            new_result_dict["pass_at_k_value"] = int(pass_at_k)
            new_result_dict["original_sample_idx"] = original_sample_idx
            new_result_dict["repeat_idx"] = i % k
            new_result_dict["k_value"] = k

            result_dicts_with_passk.append(new_result_dict)

        return result_dicts_with_passk

    def _compute_metrics_with_passk(self, sample_result_dicts, k):
        """Compute metrics including pass@k statistics."""
        # 聚合各个 data source 的 reward
        data_source_reward_dict = {}
        data_source_passk_dict = {}

        for result in sample_result_dicts:
            data_source = result["data_source"]
            data_source_reward_dict.setdefault(data_source, {})
            data_source_passk_dict.setdefault(data_source, [])

            # Regular reward metrics
            for key, value in result.items():
                if "reward" in key and value is not None:
                    data_source_reward_dict[data_source].setdefault(key, []).append(
                        value
                    )

            # Collect pass@k information (only count each original sample once)
            if (
                result.get("repeat_idx", 0) == 0
            ):  # Only count the first repeat of each original sample
                data_source_passk_dict[data_source].append(
                    result.get("pass_at_k_value", 0)
                )

        # 生成 metric_dict
        metric_dict = {}
        data_split = "val"

        # Regular metrics
        for data_source, source_rewards in data_source_reward_dict.items():
            for reward_name, rewards in source_rewards.items():
                data_source_clean = data_source.replace("/", "_")
                reward_name_clean = reward_name.replace("/", "_")
                key = f"{data_split}_{reward_name_clean}/{data_source_clean}"
                metric_dict[key] = np.mean(rewards)

        # Pass@k metrics
        for data_source, passk_values in data_source_passk_dict.items():
            if passk_values:  # Only add if we have data
                data_source_clean = data_source.replace("/", "_")
                passk_key = (
                    f"{data_split}_pass_at_{k}_accuracy_reward/{data_source_clean}"
                )
                metric_dict[passk_key] = np.mean(passk_values)
                print(
                    f"Pass@{k} accuracy for {data_source}: {np.mean(passk_values):.4f} ({np.mean(passk_values)*100:.2f}%)"
                )

        return metric_dict

    def evaluate(self, dataset: Dataset | None = None) -> dict:
        """Run evaluation on the given dataset.

        Args:
            dataset: Dataset to evaluate on. If None, uses the dataset from initialization.

        Returns:
            dict: Evaluation metrics
        """
        if dataset is not None:
            # Update validation dataset if a new one is provided
            self.val_dataset = dataset
            # Recreate dataloader with new dataset
            self._create_dataloader(
                self.train_dataset, dataset, self.collate_fn, self.train_sampler
            )

        print(f"Starting evaluation on {len(self.val_dataset)} samples...")
        self.global_steps = 0
        self.total_training_steps = 1

        # Use the inherited _validate method from RayPPOTrainer
        val_metrics = self._validate()

        # Format and display evaluation results
        self._display_evaluation_results(val_metrics)

        # Save evaluation results to file
        if self.config.output_dir is None:
            self._save_evaluation_results(val_metrics)
        else:
            print(f"=====DEBUG=====: output_dir: {self.config.output_dir}")
            self._save_evaluation_results(val_metrics, self.config.output_dir)

        return val_metrics

    def _display_evaluation_results(self, val_metrics: dict):
        """Display evaluation results grouped by data_source."""
        from datetime import datetime

        print("\n" + "=" * 80)
        print("🎯 EVALUATION RESULTS")
        print("=" * 80)

        # Convert numpy types to Python native types for better display
        formatted_metrics = {}
        for key, value in val_metrics.items():
            if hasattr(value, "item"):  # numpy types
                formatted_metrics[key] = value.item()
            else:
                formatted_metrics[key] = value

        # Group metrics by data_source
        data_sources = {}
        auxiliary_metrics = {}

        for key, value in formatted_metrics.items():
            if "/" in key and not key.startswith("val-aux"):
                # Extract data_source from metric key like 'val_accuracy_reward/vstar_bench_tool_agent'
                metric_type, data_source = key.split("/", 1)
                if data_source not in data_sources:
                    data_sources[data_source] = {}
                data_sources[data_source][metric_type] = value
            elif key.startswith("val-aux"):
                # Auxiliary metrics like 'val-aux/num_turns/min'
                auxiliary_metrics[key] = value

        # Display results for each data_source
        for data_source, metrics in data_sources.items():
            print(f"\n📊 {data_source.upper()}")
            print("-" * 60)

            # Group by metric type
            reward_metrics = {}
            other_metrics = {}

            for metric_key, value in metrics.items():
                clean_key = metric_key.replace("val_", "")
                if "reward" in clean_key:
                    reward_metrics[clean_key] = value
                else:
                    other_metrics[clean_key] = value

            # Display reward metrics first
            if reward_metrics:
                print("  💰 Reward Metrics:")
                for key, value in reward_metrics.items():
                    display_key = key.replace("_reward", "").replace("_", " ").title()
                    # Special formatting for pass@k metrics
                    if "pass_at_" in key:
                        display_key = (
                            key.replace("_reward", "")
                            .replace("_", "@", 1)
                            .replace("_", " ")
                            .title()
                        )
                        display_key = display_key.replace(
                            "@", "@"
                        )  # Ensure @ symbol is preserved

                    if isinstance(value, float):
                        formatted_value = f"{value:.4f} ({value*100:.2f}%)"
                    else:
                        formatted_value = str(value)
                    print(f"    {display_key:<25} : {formatted_value}")

            # Display other metrics
            if other_metrics:
                print("  📈 Other Metrics:")
                for key, value in other_metrics.items():
                    display_key = key.replace("_", " ").title()
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    print(f"    {display_key:<25} : {formatted_value}")

        # Display auxiliary metrics
        if auxiliary_metrics:
            print(f"\n🔄 AUXILIARY METRICS")
            print("-" * 60)
            for key, value in auxiliary_metrics.items():
                # Clean up the key for display
                display_key = key.replace("val-aux/", "").replace("/", " | ")
                print(f"  {display_key:<35} : {value}")

        print("\n" + "=" * 80)
        print(
            f"⏰ Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 80 + "\n")

    def _save_evaluation_results(self, val_metrics: dict, output_dir: Optional[str]=None):
        """Save evaluation results to a structured directory with config and results."""
        import json
        import os
        from datetime import datetime

        # Convert numpy types to JSON serializable types
        serializable_metrics = {}
        for key, value in val_metrics.items():
            if hasattr(value, "item"):  # numpy types
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = value

        passk = self.config.actor_rollout_ref.rollout.val_kwargs.n

        # Extract bench information from data sources
        bench_names = self._extract_bench_names(val_metrics)
        bench_str = "_".join(sorted(bench_names)) if bench_names else "unknown"

        # Extract model info from config path
        model_path = getattr(self.config.actor_rollout_ref.model, "path", "")
        model_info = self._extract_model_info(model_path)

        # Create timestamp and directory name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"eval_{timestamp}_{bench_str}_{model_info}_pass@{passk}"
        if output_dir is None or output_dir=="/verl_vision/evaluation_results":
            results_dir = os.path.join("/verl_vision/evaluation_results", dir_name)
        else:
            results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)

        # Save config as YAML
        config_path = os.path.join(results_dir, "config.yaml")
        self._save_config_yaml(config_path)

        # Group results by data_source for structured saving
        grouped_results = self._group_results_by_data_source(serializable_metrics)

        # Prepare the full results dictionary
        results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_config": {
                "dataset_size": (
                    len(self.val_dataset) if hasattr(self, "val_dataset") else None
                ),
                "device": getattr(self, "device_name", "unknown"),
                "bench_names": list(bench_names),
            },
            "raw_metrics": serializable_metrics,
            "grouped_by_data_source": grouped_results["data_sources"],
            "auxiliary_metrics": grouped_results["auxiliary"],
            "summary": self._generate_summary_by_data_source(
                grouped_results["data_sources"]
            ),
        }

        # Save results to JSON file
        results_path = os.path.join(results_dir, "evaluation_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to directory: {results_dir}")
        print(f"  📄 Config: {config_path}")
        print(f"  📊 Results: {results_path}")

        # Save all trajectories if we have the data
        if hasattr(self, "_last_test_batch") and hasattr(
            self, "_last_sample_result_dicts"
        ):
            trajectory_dir = self._save_all_trajectories(
                self._last_test_batch, self._last_sample_result_dicts, results_dir
            )
            if trajectory_dir:
                print(f"  🎆 Trajectories: {trajectory_dir}")

        # Also create a symlink to latest for easy access
        latest_link = os.path.join("evaluation_results", "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            import shutil

            shutil.rmtree(latest_link)

        try:
            os.symlink(dir_name, latest_link)
            print(f"🔗 Latest results linked at: {latest_link}")
        except OSError:
            # Fallback for systems that don't support symlinks
            print(f"💾 Latest results available at: {results_dir}")

        return results_dir

    def _group_results_by_data_source(self, metrics: dict) -> dict:
        """Group metrics by data_source."""
        data_sources = {}
        auxiliary_metrics = {}

        for key, value in metrics.items():
            if "/" in key and not key.startswith("val-aux"):
                metric_type, data_source = key.split("/", 1)
                if data_source not in data_sources:
                    data_sources[data_source] = {}
                data_sources[data_source][metric_type] = value
            elif key.startswith("val-aux"):
                auxiliary_metrics[key] = value

        return {"data_sources": data_sources, "auxiliary": auxiliary_metrics}

    def _generate_summary_by_data_source(self, data_sources: dict) -> dict:
        """Generate a summary organized by data_source."""
        summary = {}

        for data_source, metrics in data_sources.items():
            summary[data_source] = {}

            for metric_key, value in metrics.items():
                if "pass_at_" in metric_key and "accuracy_reward" in metric_key:
                    # Extract k value from metric key like 'val_pass_at_5_accuracy_reward'
                    k_part = metric_key.split("pass_at_")[1].split("_")[0]
                    summary[data_source][
                        f"pass@{k_part}_accuracy"
                    ] = f"{value:.4f} ({value*100:.2f}%)"
                elif "accuracy_reward" in metric_key and "pass_at_" not in metric_key:
                    summary[data_source]["accuracy"] = f"{value:.4f} ({value*100:.2f}%)"
                elif "format_reward" in metric_key:
                    summary[data_source]["format"] = f"{value:.4f} ({value*100:.2f}%)"
                elif "final_reward" in metric_key:
                    summary[data_source]["final"] = f"{value:.4f} ({value*100:.2f}%)"

        return summary

    def _extract_bench_names(self, val_metrics: dict) -> set:
        """Extract bench names from validation metrics."""
        bench_names = set()

        for key in val_metrics.keys():
            if "/" in key and not key.startswith("val-aux"):
                # Extract data_source from metric key like 'val_accuracy_reward/vstar_bench_tool_agent'
                _, data_source = key.split("/", 1)
                # Extract bench name (remove common suffixes like _tool_agent, _bench, etc.)
                bench_name = (
                    data_source.replace("_bench", "")
                    .replace("_tool_agent", "")
                    .replace("_agent", "")
                )
                # Further clean up
                bench_name = (
                    bench_name.split("_")[0] if "_" in bench_name else bench_name
                )
                bench_names.add(bench_name)

        return bench_names

    def _extract_model_info(self, model_path: str) -> str:
        """Extract model info from model path for directory naming.

        Examples:
        - '/verl_exp/tool_001rew_thyme_maxpx4096/global_step_80/actor/huggingface'
          -> 'tool_001rew_thyme_maxpx4096_step80'
        - '/verl_model/Qwen2.5-VL-3B-Instruct' -> 'Qwen2.5-VL-3B-Instruct'
        - '/path/to/model/checkpoint-1000' -> 'checkpoint-1000'
        """
        if not model_path:
            return "unknown_model"

        # Split path into components
        path_parts = model_path.strip("/").split("/")

        # Check if this is a standard model path (like /verl_model/ModelName)
        for i, part in enumerate(path_parts):
            if part in ["verl_model", "models", "huggingface_models"]:
                # Return the model name (next part after the models directory)
                if i + 1 < len(path_parts):
                    return path_parts[i + 1]

        # Look for experiment name and step info (for fine-tuned models)
        exp_name = None
        step_info = None

        for part in path_parts:
            # Extract experiment name (usually contains underscore patterns)
            if "_" in part and any(
                keyword in part for keyword in ["tool", "exp", "rew", "thyme"]
            ):
                exp_name = part
            # Extract step information
            elif "global_step_" in part:
                step_num = part.replace("global_step_", "")
                step_info = f"step{step_num}"
            elif "checkpoint-" in part:
                step_info = part

        # Construct model info string for fine-tuned models
        if exp_name and step_info:
            return f"{exp_name}_{step_info}"
        elif exp_name:
            return exp_name
        elif step_info:
            return step_info
        else:
            # Fallback: use last meaningful directory name
            meaningful_parts = [
                p for p in path_parts if p not in ["actor", "huggingface", "model"]
            ]
            return meaningful_parts[-1] if meaningful_parts else "unknown_model"

    def _save_config_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        import yaml
        from omegaconf import OmegaConf

        try:
            # Convert config to dict if it's OmegaConf
            if hasattr(self.config, "_content"):
                config_dict = OmegaConf.to_container(
                    self.config.actor_rollout_ref, resolve=True
                )
            else:
                config_dict = self.config.actor_rollout_ref

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_dict,
                    f,
                    indent=2,
                    default_flow_style=False,
                    allow_unicode=True,
                )

        except Exception as e:
            print(f"⚠️  Warning: Could not save config to YAML: {e}")
            # Fallback: save config as string representation
            with open(config_path.replace(".yaml", ".txt"), "w", encoding="utf-8") as f:
                f.write(str(self.config))

    def _save_all_trajectories(self, test_batch, sample_result_dicts, results_dir: str):
        """Save all trajectories to the evaluation results directory."""
        import time

        import torch

        from recipe.med.trajectory_saver import (extract_text_data,
                                                save_trajectories_jsonl)
        from verl import DataProto

        print("\n🚀 Saving all trajectories...")
        start_time = time.time()

        # Calculate response length for all samples if not already done
        if "response_length" not in test_batch.batch:
            batched_response_length = {
                "response_length": torch.sum(test_batch.batch["response_mask"], dim=1)
            }
            batched_response_length = DataProto.from_dict(batched_response_length)
            test_batch = test_batch.union(batched_response_length)

        # Create trajectories directory within the evaluation results directory
        trajectories_dir = os.path.join(results_dir, "trajectories")
        os.makedirs(trajectories_dir, exist_ok=True)

        # Group samples by data_source
        from collections import defaultdict

        data_source_groups = defaultdict(list)

        exp_name = self.config.trainer.get("experiment_name", "evaluation")
        step = getattr(self, "global_steps", 0)

        print(f"Processing {len(test_batch)} samples...")

        # First pass: group by data_source and generate trajectories
        all_trajectories = []
        all_samples = []
        all_result_dicts = []

        for i, sample in enumerate(test_batch):
            # Get data_source from result_dicts
            data_source = sample_result_dicts[i].get("data_source", "unknown")

            # Use the existing extract_text_data function which handles images
            save_images = self.config.get("save_images", False)  # Default to False
            trajectory = extract_text_data(
                sample,
                self.tokenizer,
                trajectories_dir,  # Use trajectories_dir as dump_path
                exp_name,
                step,
                save_images,
            )

            # Group by data_source
            data_source_groups[data_source].append(
                {
                    "trajectory": trajectory,
                    "sample": sample,
                    "result_dict": sample_result_dicts[i],
                    "index": i,
                }
            )

            all_trajectories.append(trajectory)
            all_samples.append(sample)
            all_result_dicts.append(sample_result_dicts[i])

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_batch)} samples")

        if all_trajectories:
            print(f"\\n📁 Saving trajectories by data_source...")
            saved_files = []

            # Save trajectories for each data_source separately
            for data_source, group_data in data_source_groups.items():
                data_source_clean = data_source.replace("/", "_").replace(" ", "_")

                trajectories_for_source = [item["trajectory"] for item in group_data]
                samples_for_source = [item["sample"] for item in group_data]
                result_dicts_for_source = [item["result_dict"] for item in group_data]

                # Save JSONL file for this data_source
                jsonl_file = save_trajectories_jsonl(
                    trajectories_for_source,
                    samples_for_source,
                    trajectories_dir,
                    data_source_clean,  # Use data_source as category name
                    result_dicts_for_source,
                )

                saved_files.append(jsonl_file)
                print(
                    f"  💾 {data_source}: {len(trajectories_for_source)} samples -> {os.path.basename(jsonl_file)}"
                )

            print(f"\\n📊 Summary:")
            print(f"  Total samples processed: {len(all_trajectories)}")
            print(f"  Data sources: {len(data_source_groups)}")
            print(f"  Files saved: {len(saved_files)}")
            print(f"  Images and trajectory data saved in: {trajectories_dir}")

            # Generate summary statistics
            self._print_trajectory_summary(all_samples, all_result_dicts)

        else:
            print("⚠️  No trajectories were generated")

        end_time = time.time()
        print(f"⏱️  Trajectory saving completed in {end_time - start_time:.2f}s\n")

        return trajectories_dir if all_trajectories else None

    def _print_trajectory_summary(self, samples_list, result_dicts_list):
        """Print a summary of saved trajectories."""
        from collections import Counter

        import torch

        print("\n📊 Trajectory Summary:")
        print("=" * 50)

        if not samples_list:
            return

        # Response length statistics
        response_lengths = [
            (
                int(sample.batch["response_length"].item())
                if torch.is_tensor(sample.batch["response_length"])
                else int(sample.batch["response_length"])
            )
            for sample in samples_list
        ]

        print(
            f"  Response Lengths: min={min(response_lengths)}, max={max(response_lengths)}, avg={sum(response_lengths)/len(response_lengths):.1f}"
        )

        # Tool usage statistics
        tool_counts = [
            sample.non_tensor_batch.get("tool_call_counts", 0)
            for sample in samples_list
        ]
        with_tools = sum(1 for count in tool_counts if count > 0)
        print(f"  Tool Usage: {with_tools}/{len(samples_list)} samples used tools")

        # Accuracy statistics
        if result_dicts_list:
            correct = sum(
                1
                for result_dict in result_dicts_list
                if result_dict.get("accuracy_reward", 0) > 0
            )
            accuracy_rate = correct / len(result_dicts_list) * 100
            print(
                f"  Accuracy: {correct}/{len(result_dicts_list)} ({accuracy_rate:.1f}%)"
            )

            # Data source distribution
            data_sources = [
                result_dict.get("data_source", "unknown")
                for result_dict in result_dicts_list
            ]
            source_counts = Counter(data_sources)
            print(f"  Data Sources: {dict(source_counts)}")

        print("=" * 50)

    def _should_enable_llm_verification(self, result_dicts: list) -> bool:
        """Determine if LLM verification should be enabled based on config and data sources."""
        llm_verification = self.config.get("llm_verification", {})
        enabled = llm_verification.get("enabled", False)

        if enabled == "auto":
            # Auto enable for charxiv and olympiadbench datasets
            for result_dict in result_dicts:
                data_source = result_dict.get("data_source", "")
                if "charxiv" in data_source.lower() or "olympiadbench" in data_source.lower():
                    return True
            return False
        elif enabled is True:
            return True
        else:
            return False

    def _llm_verify_accuracy(self, batch: DataProto, result_dicts: list) -> list:
        """Use LLM to verify the accuracy of responses and update result_dicts."""
        import time

        llm_verification = self.config.get("llm_verification", {})
        verifier_model_path = llm_verification.get("verifier_model", {}).get("path", "")

        if not verifier_model_path:
            print(
                "Warning: verifier_model.path not configured, skipping LLM verification"
            )
            return result_dicts

        print(f"Switching to verification model: {verifier_model_path}")
        start_time = time.time()

        # Switch vLLM engine to verification model
        verified_result_dicts = self._llm_verification(
            result_dicts, verifier_model_path
        )

        end_time = time.time()
        print(f"LLM verification completed in {end_time - start_time:.2f}s")

        return verified_result_dicts

    def _llm_verification(self, result_dicts: list, verifier_model_path: str) -> list:
        """LLM verification implementation."""
        # Step 1: Kill current VLM model to free memory
        print("Killing Current vllm engine to free memory...")
        self._kill_current_model()

        # Step 2: Initialize verification engine with the 32B LLM
        print(f"Loading verification model: {verifier_model_path}")
        self._init_verification_engine(verifier_model_path)

        # Step 3: Perform batch verification
        print(f"🔍 Verifying {len(result_dicts)} responses...")
        verified_result_dicts = self._batch_verify_responses(result_dicts)

        # Step 4: Cleanup verification engine
        print("🧹 Cleaning up verification engine...")
        self._cleanup_verification_engine()

        return verified_result_dicts

    def _kill_current_model(self):
        """Kill the current VLM model and free Ray GPU resources."""
        # Step 1: Kill rollout workers
        if hasattr(self, "actor_rollout_wg"):
            print("  Killing rollout worker group...")
            del self.actor_rollout_wg

        if hasattr(self, "async_rollout_manager"):
            print("  Killing async rollout manager...")
            del self.async_rollout_manager

        # Step 2: Remove placement groups to free GPU resources
        import ray

        print("  Removing Ray placement groups...")
        try:
            # Get all placement groups and remove them
            placement_groups = ray.util.placement_group_table()
            for pg_id, pg_info in placement_groups.items():
                if pg_info["state"] == "CREATED":
                    print(f"    Removing placement group: {pg_id}")
                    group = ray.util.get_placement_group(pg_info["name"])
                    # Use placement group ID directly instead of name
                    ray.util.remove_placement_group(group)
        except Exception as pg_e:
            print(f"  Warning: Failed to remove some placement groups: {pg_e}")

        # Step 3: Clear resource pool to free GPU resources
        if hasattr(self, "resource_pool_manager"):
            print("  Clearing resource pool...")
            try:
                # Clear the resource pool dict to release GPU resources
                self.resource_pool_manager.resource_pool_dict.clear()
            except Exception as pool_e:
                print(f"  Warning: Failed to clear resource pool: {pool_e}")

        # Step 4: Force garbage collection and clear GPU memory
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        print("  Memory and GPU resources cleared")

    def _init_verification_engine(self, verifier_model_path: str):
        """Initialize the verification engine with 32B LLM (Ray Actor or direct vLLM/SGLang)."""
        llm_verification = self.config.get("llm_verification", {})
        verifier_model = llm_verification.get("verifier_model", {})
        max_tokens = int(verifier_model.get("max_tokens", 128))
        
        use_ray_actor = verifier_model.get("use_ray_actor", True)
        backend = verifier_model.get("backend", "vllm")  # vllm or sglang
        print(f"USE_RAY_ACTOR: {use_ray_actor}, BACKEND: {backend}")
        print(verifier_model)
        
        verification_config = {
            "model": verifier_model_path,
            "tensor_parallel_size": verifier_model.get("tensor_parallel_size", 1),
            "gpu_memory_utilization": verifier_model.get("gpu_memory_utilization", 0.8),
            "max_model_len": verifier_model.get("max_model_len", 4096),
            "dtype": verifier_model.get("dtype", "bfloat16"),
            "trust_remote_code": True,
            "enforce_eager": False,
            "max_tokens": max_tokens,
        }

        if use_ray_actor:
            # Use Ray Actor for verification LLM
            if backend == "sglang":
                self.verification_llm_actor = VerificationLLMActorSGLang.remote(verification_config)
                self.verification_mode = "ray_actor_sglang"
                print("✅ Verification engine (Ray Actor + SGLang) initialized")
            else:
                self.verification_llm_actor = VerificationLLMActor.remote(verification_config)
                self.verification_mode = "ray_actor_vllm"
                print("✅ Verification engine (Ray Actor + vLLM) initialized")
        else:
            # Use direct vLLM/SGLang for verification
            if backend == "sglang":
                import sglang as sgl
                
                self.verification_llm = sgl.Engine(
                    model_path=verification_config["model"],
                    tp_size=verification_config["tensor_parallel_size"],
                    mem_fraction_static=verification_config["gpu_memory_utilization"],
                    context_length=verification_config["max_model_len"],
                    dtype=verification_config["dtype"],
                    trust_remote_code=verification_config["trust_remote_code"],
                )
                
                self.verification_sampling_params = {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_new_tokens": verification_config["max_tokens"],
                }
                
                self.verification_mode = "direct_sglang"
                print("✅ Verification engine (direct SGLang) initialized")
            else:
                from vllm import LLM, SamplingParams
                
                self.verification_llm = LLM(
                    model=verification_config["model"],
                    tensor_parallel_size=verification_config["tensor_parallel_size"],
                    gpu_memory_utilization=verification_config["gpu_memory_utilization"],
                    max_model_len=verification_config["max_model_len"],
                    dtype=verification_config["dtype"],
                    trust_remote_code=verification_config["trust_remote_code"],
                    enforce_eager=verification_config["enforce_eager"],
                )
                
                self.verification_sampling_params = SamplingParams(
                    temperature=0.0,  # Deterministic judgment
                    top_p=1.0,
                    max_tokens=verification_config["max_tokens"],
                )
                
                self.verification_mode = "direct_vllm"
                print("✅ Verification engine (direct vLLM) initialized")

    def _batch_verify_responses(self, result_dicts: list) -> list:
        """Perform batch verification of responses using the verification engine."""
        print(f"  Creating verification prompts for {len(result_dicts)} responses...")

        # Step 1: Collect all verification prompts
        verification_prompts = []
        for result_dict in result_dicts:
            verification_prompt = self._create_verification_prompt(result_dict)
            verification_prompts.append(verification_prompt)

        print(
            f"  Running batch LLM inference on {len(verification_prompts)} prompts..."
        )

        try:
            # Step 2: Get all LLM judgments in batch
            verification_responses = self._get_llm_judgment_batch(verification_prompts)

            # Step 3: Parse responses and update result_dicts
            verified_result_dicts = []
            for i, (result_dict, verification_response) in enumerate(
                zip(result_dicts, verification_responses)
            ):
                try:
                    updated_result_dict = self._parse_verification_response(
                        result_dict, verification_response
                    )
                    verified_result_dicts.append(updated_result_dict)
                except Exception as e:
                    print(
                        f"⚠️  Warning: Failed to parse verification for sample {i}: {e}"
                    )
                    verified_result_dicts.append(result_dict)

            print(
                f"  ✅ Batch verification completed: {len(verified_result_dicts)} responses processed"
            )
            return verified_result_dicts

        except Exception as e:
            print(f"⚠️  Error: Batch verification failed: {e}")
            # Fallback: return original result_dicts
            return result_dicts

    def _create_verification_prompt(self, result_dict: dict) -> str:
        """Create a prompt for LLM to verify the accuracy of a response."""
        question = result_dict.get("question", "")
        response = result_dict.get("response", "")
        answer = result_dict.get("answer", "")
        id = result_dict.get("id", "")
        data_source = result_dict.get("data_source", "")

        if "charxiv2rq" in data_source:
            from recipe.med.eval.utils.charxiv.constants import (
                REASONING_GRADING_INST, REASONING_GRADING_PREFIX)

            reasoning_q_source = int(id.split("_")[-2])
            # get query for answer type (inst_category), then
            # populate the query with the question, ground truth, and response
            verification_prompt = REASONING_GRADING_PREFIX + REASONING_GRADING_INST[
                reasoning_q_source
            ].replace("<|question|>", question).replace(
                "<|ground_truth|>", answer
            ).replace(
                "<|response|>", response
            )
        else:
            from recipe.med.eval.utils.common import PROMPT_JUDGE
            
            # Use general verification prompt for other benchmarks
            verification_prompt = PROMPT_JUDGE.format(
                question=question,
                answer=answer,
                response=response
            )


        verification_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": verification_prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        return verification_prompt

    def _get_llm_judgment(self, prompt: str) -> str:
        """Get judgment from the verification LLM (single prompt)."""
        if self.verification_mode in ["ray_actor_vllm", "ray_actor_sglang"]:
            # Generate response using Ray Actor
            content = ray.get(self.verification_llm_actor.generate_single.remote(prompt))
            return content
        elif self.verification_mode == "direct_vllm":
            # Generate response using direct vLLM
            outputs = self.verification_llm.generate([prompt], self.verification_sampling_params)
            return outputs[0].outputs[0].text.strip()
        elif self.verification_mode == "direct_sglang":
            # Generate response using direct SGLang
            output = self.verification_llm.generate([prompt], self.verification_sampling_params)
            return output[0]["text"].strip()
        else:
            raise ValueError(f"Unknown verification mode: {self.verification_mode}")

    def _get_llm_judgment_batch(self, prompts: list[str]) -> list[str]:
        """Get judgments from the verification LLM (batch prompts)."""
        if self.verification_mode in ["ray_actor_vllm", "ray_actor_sglang"]:
            # Generate responses using Ray Actor for all prompts at once
            responses = ray.get(self.verification_llm_actor.generate_batch.remote(prompts))
            return responses
        elif self.verification_mode == "direct_vllm":
            # Generate responses using direct vLLM
            outputs = self.verification_llm.generate(prompts, self.verification_sampling_params)
            responses = []
            for output in outputs:
                content = output.outputs[0].text.strip()
                responses.append(content)
            return responses
        elif self.verification_mode == "direct_sglang":
            # Generate responses using direct SGLang
            outputs = self.verification_llm.generate(prompts, self.verification_sampling_params)
            return [output["text"].strip() for output in outputs]
        else:
            raise ValueError(f"Unknown verification mode: {self.verification_mode}")

    def _parse_verification_response(
        self, original_result_dict: dict, verification_response: str
    ) -> dict:
        """Parse verification response and update result_dict accordingly."""
        result_dict = original_result_dict.copy()
        data_source = result_dict.get("data_source", "")

        # Parse LLM judgment based on data source
        if "charxiv2rq_bench" in data_source:
            # Parse JSON format response for charxiv
            import json
            import re
            
            try:
                # Try to extract score from JSON response
                score_match = re.search(r'"score"\s*:\s*([01])', verification_response)
                if score_match:
                    llm_accuracy = float(int(score_match.group(1)))
                else:
                    # Fallback: try full JSON parsing
                    try:
                        parsed = json.loads(verification_response.strip())
                        score = parsed.get("score", 0)
                        llm_accuracy = float(score)
                    except json.JSONDecodeError:
                        llm_accuracy = 0.0
            except Exception:
                llm_accuracy = 0.0
        else:
            # Parse JSON format response for general benchmarks using PROMPT_JUDGE
            import json
            import re
            
            try:
                # Try to extract score from JSON response
                score_match = re.search(r'"score"\s*:\s*([01])', verification_response)
                if score_match:
                    llm_accuracy = float(int(score_match.group(1)))
                else:
                    # Fallback: try full JSON parsing
                    try:
                        parsed = json.loads(verification_response.strip())
                        score = parsed.get("score", 0)
                        llm_accuracy = float(score)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try CORRECT/INCORRECT format as last resort
                        verification_response_upper = verification_response.upper().strip()
                        llm_accuracy = 1.0 if "CORRECT" in verification_response_upper else 0.0
            except Exception:
                llm_accuracy = 0.0

        # Update accuracy_reward based on LLM judgment
        original_accuracy = result_dict.get("accuracy_reward", 0.0)

        # Store both original and LLM judgments
        result_dict["accuracy_reward_original"] = original_accuracy
        result_dict["accuracy_reward_llm"] = llm_accuracy
        result_dict["accuracy_reward"] = int(llm_accuracy) | int(original_accuracy)
        result_dict["llm_verification_response"] = verification_response

        # Update final_reward if it exists
        if "final_reward" in result_dict:
            format_reward = result_dict.get("format_reward", 0.0)
            result_dict["final_reward"] = format_reward + llm_accuracy

        return result_dict

    def _cleanup_verification_engine(self):
        """Cleanup the verification engine to free memory."""
        try:
            if self.verification_mode in ["ray_actor_vllm", "ray_actor_sglang"]:
                if hasattr(self, "verification_llm_actor"):
                    # Cleanup Ray Actor resources
                    ray.get(self.verification_llm_actor.cleanup.remote())
                    # Kill the actor
                    ray.kill(self.verification_llm_actor)
                    del self.verification_llm_actor
                    backend_name = "vLLM" if self.verification_mode == "ray_actor_vllm" else "SGLang"
                    print(f"  Verification engine (Ray Actor + {backend_name}) cleaned up")
            elif self.verification_mode == "direct_vllm":
                if hasattr(self, "verification_llm"):
                    # Cleanup direct vLLM resources
                    del self.verification_llm
                if hasattr(self, "verification_sampling_params"):
                    del self.verification_sampling_params
                print("  Verification engine (direct vLLM) cleaned up")
            elif self.verification_mode == "direct_sglang":
                if hasattr(self, "verification_llm"):
                    # Cleanup direct SGLang resources
                    self.verification_llm.shutdown()
                    del self.verification_llm
                if hasattr(self, "verification_sampling_params"):
                    del self.verification_sampling_params
                print("  Verification engine (direct SGLang) cleaned up")

            # Force garbage collection
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Failed to cleanup verification engine: {e}")
