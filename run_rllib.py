from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ars import ARSTrainer
from ray.rllib.agents.a3c import A2CTrainer

import gym
import pybullet_envs
from ray.tune.registry import register_env

def bullet_walker_creator(env_config):
    import pybullet_envs

    return gym.make("Walker2DBulletEnv-v0")

def bullet_cheetah_creator(env_config):
    import pybullet_envs

    return gym.make("HalfCheetahBulletEnv-v0")

def bullet_hopper_creator(env_config):
    import pybullet_envs

    return gym.make("HopperBulletEnv-v0")

register_env("Walker2DBulletEnv-v0", bullet_walker_creator)
register_env("HalfCheetahBulletEnv-v0", bullet_cheetah_creator)
register_env("HopperBulletEnv-v0", bullet_hopper_creator)


env_names = tune.grid_search(["Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0"])
on_policy_trainers = [PPOTrainer, ARSTrainer, A2CTrainer]


tune.run(on_policy_trainers, config={"env": env_names, "batch_mode":"complete_episodes"}, checkpoint_at_end=True)
