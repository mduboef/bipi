# Step-by-step plan

# set up DeepSeaTreasure-v0 environment
    # 11×10 episodic grid-world
    # The agent starts at (0,0), episdoe ends when treaure is found at (10,0) or after 100 steps
    # Four actions (up/down/left/right)
    # Reward is a 2-dimensional vector: (time_penalty=-1 per step, treasure_value)

# Compute CCS (with J vectors, corner weights and policies)
    # DeepSeaTreasure-v0 known to have 10 regions/policies
    # likely a good idea to use linear support from LucasAlegre/morl-baselines
        # paired with MPMOQLearning outer-loop solver

# source demonstrations paired with underlying preference weights
    # use Boltzmann-rational synthetic demos
    # demons need to give state and action for each timestep

# implement BIPI

# implement DWPI
    # 4 different ways to represent demos (input layer to inference network):
        # based on return vector
        # based on state frequency
        # based on successor features (new/novel)
        # based on learned embedding of trajectory (new/novel but unclear how to do this)

# try more complex environments
    # consider D4MORL demonstration database for continuous-control MuJoCo envs

# test robustness of approaches to various types of misspecification
    # ie. \beta_{demo} assumed doesn't match true \beta_{demo} used to generate demos, 

import mo_gymnasium as mo_gym
import numpy as np


# action index -> human-readable label
ACTION_LABELS = {0: "up", 1: "down", 2: "left", 3: "right"}

# reward vector indices
TREASURE_IDX = 0
TIME_IDX = 1


# creates and returns the DeepSeaTreasure-v0 environment (convex map, Yang et al. 2019)
def makeEnv():
	env = mo_gym.make("deep-sea-treasure-v0")
	return env


# prints environment dimensions and the known Pareto front
def printEnvInfo(env):
	print("=== DeepSeaTreasure-v0 ===")
	print(f"  observation space : {env.observation_space}")
	print(f"  action space      : {env.action_space}  {ACTION_LABELS}")
	print(f"  reward space      : {env.unwrapped.reward_space}")
	print(f"  reward dims       : [treasure_value, time_penalty]")
	print()
	paretoFront = env.unwrapped.pareto_front(gamma=1.0)
	print(f"  known Pareto front ({len(paretoFront)} policies):")
	for i, rv in enumerate(paretoFront):
		print(f"    policy {i+1:2d}: treasure={rv[TREASURE_IDX]:.1f}, time={rv[TIME_IDX]:.0f}")
	print()


# runs one episode using the provided policy callable; returns (trajectory, totalReward)
# policy(obs, env) -> action int
def runEpisode(env, policy):
	obs, info = env.reset()
	trajectory = []
	totalReward = np.zeros(2)
	done = False

	while not done:
		action = policy(obs, env)
		nextObs, reward, terminated, truncated, info = env.step(action)
		trajectory.append((obs.copy(), action, reward.copy()))
		totalReward += reward
		obs = nextObs
		done = terminated or truncated

	return trajectory, totalReward


# uniform random policy
def randomPolicy(obs, env):
	return env.action_space.sample()


if __name__ == "__main__":
	env = makeEnv()
	printEnvInfo(env)

	print("=== Random episode ===")
	trajectory, totalReward = runEpisode(env, randomPolicy)
	print(f"  steps            : {len(trajectory)}")
	print(f"  total reward     : treasure={totalReward[TREASURE_IDX]:.1f}, time={totalReward[TIME_IDX]:.0f}")
	print(f"  final state      : {trajectory[-1][0]}")

	env.close()