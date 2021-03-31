import ray
from ray import tune
from ray.rllib.agents.trainer_template import build_trainer

from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import validate_config as ppo_validate_config
from ray.rllib.agents.ppo.ppo import execution_plan as ppo_execution_plan
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

# from ray.rllib.agents.ars import ARSTrainer
# from ray.rllib.agents.a3c import A2CTrainer

import seagul.envs
from seagul.mesh import mdim_div_stable


class PPOFracPolicy(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[sample_batch.REWARDS] = mdim_div_stable(sample_batch[sample_batch.OBS], sample_batch[sample_batch.ACTIONS], sample_batch[sample_batch.REWARDS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)


def get_ppo_policy_class(config):
    return PPOFracPolicy


PPOFracTrainer = build_trainer(
    name="PPOFrac",
    default_config=PPO_DEFAULT_CONFIG,
    default_policy=None,
    get_policy_class=get_ppo_policy_class,
)

env_names = tune.grid_search(["Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0"])
on_policy_trainers = [PPOFracTrainer]#, ARSTrainer, A2CTrainer]

tune.run(on_policy_trainers, config={"env": env_names, "batch_mode": "complete_episodes", "framework": "torch"}, checkpoint_at_end=True)
