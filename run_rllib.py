from ray import tune
from ray.rllib.agents.trainer_template import build_trainer

import seagul.envs
from seagul.mesh import mdim_div_stable

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.rllib.agents.ars import ARSTrainer

# PPO =======================================================================================
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo import validate_config as ppo_validate_config
from ray.rllib.agents.ppo.ppo import execution_plan as ppo_execution_plan
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import get_policy_class as get_ppo_policy_class


class PPOFracPolicy(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[sample_batch.REWARDS] = mdim_div_stable(sample_batch[sample_batch.OBS], sample_batch[sample_batch.ACTIONS], sample_batch[sample_batch.REWARDS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)


def get_ppo_frac_policy_class(config):
    return PPOFracPolicy

PPO_CONFIG["num_sgd_iter"] = 32
PPO_CONFIG["sgd_minibatch_size"] = 54096
PPO_CONFIG["train_batch_size"] = 65536
PPO_CONFIG["observation_filter"] = "MeanStdFilter"

PPOFracTrainer = build_trainer(
    name="PPOFrac",
    default_config=PPO_CONFIG,
    default_policy=None,
    execution_plan=ppo_execution_plan,
    validate_config=ppo_validate_config,
    get_policy_class=get_ppo_frac_policy_class,
)

PPOCTrainer = build_trainer(
    name="PPOFrac",
    default_config=PPO_CONFIG,
    default_policy=None,
    execution_plan=ppo_execution_plan,
    validate_config=ppo_validate_config,
    get_policy_class=get_ppo_policy_class,
)

# A2C =======================================================================================
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.rllib.agents.a3c.a2c import validate_config as a2c_validate_config
from ray.rllib.agents.a3c.a2c import execution_plan as a2c_execution_plan
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import get_policy_class as get_a2c_policy_class


class A2CFracPolicy(A3CTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[sample_batch.REWARDS] = mdim_div_stable(sample_batch[sample_batch.OBS], sample_batch[sample_batch.ACTIONS], sample_batch[sample_batch.REWARDS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)


def get_a2c_frac_policy_class(config):
    return A2CFracPolicy


A2C_CONFIG["num_sgd_iter"] = 32
A2C_CONFIG["sgd_minibatch_size"] = 54096
A2C_CONFIG["train_batch_size"] = 65536
A2C_CONFIG["observation_filter"] = "MeanStdFilter"


A2CFracTrainer = build_trainer(
    name="A2CFrac",
    default_config=A2C_CONFIG,
    default_policy=None,
    get_policy_class=get_a2c_policy_class,
    execution_plan=a2c_execution_plan,
    validate_config=a2c_validate_config
)

A2CCTrainer = build_trainer(
    name="A2CM",
    default_config=A2C_CONFIG,
    default_policy=None,
    get_policy_class=get_a2c_frac_policy_class,
    execution_plan=a2c_execution_plan,
    validate_config=a2c_validate_config
)

# from ray.rllib.agents.ars.ars import DEFAULT_CONFIG as A
# from ray.rllib.agents.a3c import A2CTrainer#

env_names = tune.grid_search(["Walker2d-v2", "Hopper-v2", "HalfCheetah-v2"])
fr_policy_trainers = [PPOFracTrainer, A2CFracTrainer]#, ARSTrainer, A2CTrainer]
on_policy_trainers = [PPOCTrainer, ARSTrainer, A2CCTrainer]

all_trainers = [*fr_policy_trainers, *on_policy_trainers]

print(all_trainers)
tune.run(all_trainers,
         config={"env": env_names, "batch_mode": "complete_episodes", "framework": "torch"},
         checkpoint_at_end=True,
         stop={"time_total_s": 3600})