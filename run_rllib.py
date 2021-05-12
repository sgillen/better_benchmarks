from abc import ABC

from ray import tune
from ray.rllib.agents.trainer_template import build_trainer

import pybullet_envs
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

# import torch
# torch.set_default_dtype(torch.float32)

class PPOFracPolicy(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[sample_batch.REWARDS] = mdim_div_stable(sample_batch[sample_batch.OBS], sample_batch[sample_batch.ACTIONS], sample_batch[sample_batch.REWARDS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)


def get_ppo_frac_policy_class(config):
    return PPOFracPolicy

# https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/halfcheetah-ppo.yaml
PPO_CONFIG["num_sgd_iter"] = 32
PPO_CONFIG["sgd_minibatch_size"] = 54096
PPO_CONFIG["train_batch_size"] = 65536
PPO_CONFIG["observation_filter"] = "MeanStdFilter"
PPO_CONFIG["lr"] = .0003
PPO_CONFIG["clip_param"] = .2
PPO_CONFIG["grad_clip"] = .5
PPO_CONFIG["kl_coeff"] = 1.0
PPO_CONFIG["lambda"] = .95
PPO_CONFIG["gamma"] = .99

PPOFracTrainer = build_trainer(
    name="PPOFrac",
    default_config=PPO_CONFIG,
    default_policy=None,
    execution_plan=ppo_execution_plan,
    validate_config=ppo_validate_config,
    get_policy_class=get_ppo_frac_policy_class,
)

PPOCTrainer = build_trainer(
    name="PPOC",
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
    name="A2CC",
    default_config=A2C_CONFIG,
    default_policy=None,
    get_policy_class=get_a2c_frac_policy_class,
    execution_plan=a2c_execution_plan,
    validate_config=a2c_validate_config
)

# ARS =======================================================================================
from ray.rllib.agents.ars.ars import DEFAULT_CONFIG as ARS_CONFIG
from ray.rllib.agents.ars.ars_torch_policy import ARSTorchPolicy
from ray.rllib.agents.es.es import validate_config
from ray.rllib.env.env_context import EnvContext

ARS_CONFIG["noise_stdev"] = .05
ARS_CONFIG['sgd_stepsize'] = .05


class ARSFracPolicy(ARSTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[sample_batch.REWARDS] = mdim_div_stable(sample_batch[sample_batch.OBS], sample_batch[sample_batch.ACTIONS],sample_batch[sample_batch.REWARDS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

def get_ars_frac_policy_class(config):
    return ARSFracPolicy


class ARSFracTrainer(ARSTrainer):
    _name = "ARSFrac"
    __name__ = "ARSFrac"
    _default_config = ARS_CONFIG

    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        validate_config(config)
        env_context = EnvContext(config["env_config"] or {}, worker_index=0)
        env = env_creator(env_context)

        policy_cls = get_ars_frac_policy_class(config)
        self.policy = policy_cls(env.observation_space, env.action_space, config)

class ARSCTrainer(ARSTrainer):
    _name = "ARSC"
    __name__ = "ARSC"
    _default_config = ARS_CONFIG


# main =======================================================================================

if __name__ == "__main__":
    log_dir = input("Enter a name for the run: ")
    input(f"saving in ./{log_dir}, press anything to continue: ")

    env_names = tune.grid_search(["Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0" ])
    #env_names = tune.grid_search(["Walker2d-v2", "Hopper-v2", "HalfCheetah-v2"])
    #env_names =  tune.grid_search(["Humanoid-v2"])
    fr_policy_trainers = [PPOFracTrainer]#, ARSFracTrainer]#, ARSTrainer, A2CTrainer]
    on_policy_trainers = [PPOCTrainer]#, ARSCTrainer]


    #all_trainers = [PPOCTrainer]
    all_trainers = [*fr_policy_trainers, *on_policy_trainers]
    #all_trainers = [ARSFracTrainer, ARSCTrainer]
    
    print(all_trainers)
    tune.run(all_trainers,
             config={"env": env_names, "batch_mode": "complete_episodes", "framework": "torch"},
             checkpoint_at_end=True,
             local_dir=log_dir,
             num_samples=4,
             stop={"timesteps_total": int(2e7)},
             )

    #{"time_total_s": 144000}
