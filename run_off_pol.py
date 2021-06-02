import ray
from ray import tune
from ray.rllib.agents.ddpg import TD3Trainer


# main =======================================================================================

if __name__ == "__main__":
    ray.shutdown()
    ray.init(object_store_memory=1024*1024*1024*30)

    print(ray.available_resources())
    
    log_dir = input("Enter a name for the run: ")
    input(f"saving in ./{log_dir}, press anything to continue: ")


    #    env_names = tune.grid_search(["Walker2DBulletEnv-v0", "HopperBulletEnv-v0", "HalfCheetahBulletEnv-v0" ])
    env_names = tune.grid_search(["Walker2d-v2", "Hopper-v2", "HalfCheetah-v2"])

    off_policy_trainers = TD3Trainer

    all_trainers = TD3Trainer
    
    print(all_trainers)
    tune.run(all_trainers,
             config={"env": env_names, "framework": "torch"},
             checkpoint_at_end=True,
             local_dir=log_dir,
             num_samples=4,
             reuse_actors=True,
             stop={"timesteps_total":1000000},
             )

    #{"time_total_s": 144000}

    16878074266.0
    32212254720.0
