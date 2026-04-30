import mo_gymnasium as mo_gym
import numpy as np
from config import ACTION_LABELS, TREASURE_IDX, TIME_IDX

# creates and returns the DeepSeaTreasure-v0 environment (convex map, Yang et al. 2019)
def makeEnv():
	env = mo_gym.make("deep-sea-treasure-v0")
	return env


# prints environment dimensions and the known Pareto front
def printCCSInfo(env, envName):
	print(f"=== {envName} ===")
	print(f"  observation space : {env.observation_space}")
	print(f"  action space      : {env.action_space}  {ACTION_LABELS}")
	print(f"  reward space      : {env.unwrapped.reward_space}")
	print(f"  reward dims       : [treasure_value, time_penalty]\n")
	paretoFront = env.unwrapped.pareto_front(gamma=1.0)
	printRegionsInfo(paretoFront)


# computes preference-space region boundaries for each policy on the Pareto front
# returns list of (returnVec, wLeft, wRight) sorted by increasing w
def computeRegions(paretoFront):
	# sort by treasure ascending — lower treasure preferred at w=0, higher at w=1
	sorted_ = sorted(paretoFront, key=lambda rv: rv[TREASURE_IDX])

	breakpoints = [0.0]
	for i in range(len(sorted_) - 1):
		r0i, r1i = sorted_[i][TREASURE_IDX],  sorted_[i][TIME_IDX]
		r0j, r1j = sorted_[i+1][TREASURE_IDX], sorted_[i+1][TIME_IDX]
		# solve w*·r0i + (1-w*)·r1i = w*·r0j + (1-w*)·r1j
		wStar = (r1j - r1i) / ((r0i - r0j) + (r1j - r1i))
		breakpoints.append(wStar)
	breakpoints.append(1.0)

	regions = []
	for i, rv in enumerate(sorted_):
		regions.append((rv, breakpoints[i], breakpoints[i+1]))
	return regions


# prints each policy's preference region and its length
def printRegionsInfo(paretoFront):
	regions = computeRegions(paretoFront)
	print(f"  preference regions (w = weight on treasure, 1-w on time):")
	print(f"  {'policy':>8}  {'treasure':>10}  {'time':>6}  {'w_left':>8}  {'w_right':>8}  {'length':>8}")
	for rv, wLeft, wRight in regions:
		print(f"  {'':>8}  {rv[TREASURE_IDX]:>10.1f}  {rv[TIME_IDX]:>6.0f}  {wLeft:>8.4f}  {wRight:>8.4f}  {wRight - wLeft:>8.4f}")
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
