import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.trainer.main_ppo import create_rl_dataset

from recipe.med.eval.ray_evaluator import RayEvaluator


class EvalTaskRunner:
    """Ray remote class for executing distributed evaluation tasks.

    This class encapsulates the main evaluation logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        from verl.workers.fsdp_workers import (ActorRolloutRefWorker,
                                               AsyncActorRolloutRefWorker)

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )
        ray_worker_group_cls = RayWorkerGroup

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Rollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.Rollout] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def run(self, config):
        """Execute the main evaluation workflow.

        This method sets up the distributed evaluation environment, initializes
        workers and datasets, then starts the evaluation process.

        Args:
            config: Evaluation configuration object containing all parameters needed
                   for setting up and running the evaluation process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        assert processor is not None

        _, ray_worker_group_cls = self.add_actor_rollout_worker(config)

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            is_training=False,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)


        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("val_max_samples", -1),
        )
        print(f"Loaded evaluation dataset with {len(val_dataset)} samples")
        print("Sample data:", val_dataset[0])
        
        # Create evaluator
        evaluator = RayEvaluator(
            config=config,
            tokenizer=tokenizer,
            val_dataset=val_dataset,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
        )
        
        # Initialize the workers of the evaluator
        evaluator.init_workers()
        
        # Run evaluation
        print("Starting evaluation...")
        val_metrics = evaluator.evaluate()  # Dataset already provided in __init__
        print("Evaluation completed!")
        
        return val_metrics


@hydra.main(config_path="config", config_name="eval", version_base=None)
def main(config):
    """Main entry point for evaluation with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing evaluation parameters.
    """
    run_eval(config)


# Define a function to run the evaluation process
def run_eval(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed evaluation process.

    Args:
        config: Evaluation configuration object containing all necessary parameters
                for distributed evaluation including Ray initialization settings,
                model paths, and evaluation hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
        from omegaconf import OmegaConf
        
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=16)(EvalTaskRunner)  # evaluation needs fewer CPUs

    # Create a remote instance of the EvalTaskRunner class
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
