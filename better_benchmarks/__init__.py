from gym.envs.registration import register

from better_benchmarks.walker2d import PBHWalker2dEnv
from better_benchmarks.hopper import PBHopperEnv
from better_benchmarks.halfcheetah import PBHalfCheetahEnv

register(id="PBWalker2d-v0", entry_point="better_benchmarks:PBHWalker2dEnv", max_episode_steps=1000)
register(id="PBHopper-v0", entry_point="better_benchmarks:PBHopperEnv", max_episode_steps=1000)
register(id="PBHalfCheetah-v0", entry_point="better_benchmarks:PBHalfCheetahEnv", max_episode_steps=1000)