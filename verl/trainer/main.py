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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json

import ray
from omegaconf import OmegaConf
import sys

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

# FORMAT_PROMPT="""Generate an image description based on the question.
# Then, provide a rationale to analyze the question.
# Next, generate a step-by-step reasoning process to solve the problem. Ensure the steps are logical and concise.
# Finally, provide a concise summary of the final answer in the following format: 'The final answer is: xxx.

# Format your response with the following sections, separated by ###:
# ### Image Description:
# ### Rationales:
# ### Let's think step by step.
# ### Step 1:
# ### Step 2:
# ...
# ### The final answer is:
# """
FORMAT_PROMPT=r"You should first thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$. If the question is a **multiple-choice question**, provide the final option letter in the format: <answer> $\boxed{option letter}$ </answer>. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\boxed{2}$ </answer>, which means your output should start with <think> and end with </answer>. The final answer MUST BE put in \boxed{}.\n"
# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)
        val_reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            prompt=FORMAT_PROMPT,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
