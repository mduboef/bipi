import time
import gymnasium
import mo_gymnasium as mo_gym
import numpy as np

# creates and returns the named mo_gymnasium environment
def makeEnv(envName):
	env = mo_gym.make(envName)
	return env


# prints environment dimensions; shows Pareto front regions if the env exposes one
def printEnvInfo(env, envName):
	print(f"=== {envName} ===")
	print(f"  observation space : {env.observation_space}")
	print(f"  action space      : {env.action_space}")
	print(f"  reward space      : {env.unwrapped.reward_space}\n")
	try:
		paretoFront = env.unwrapped.pareto_front(gamma=1.0)
		printRegionsInfo(paretoFront)
	except AttributeError:
		pass


# computes preference-space region boundaries for each policy on the Pareto front (2D only)
# returns list of (returnVec, wLeft, wRight) sorted by increasing w on obj 0
def computeRegions(paretoFront):
	sorted_ = sorted(paretoFront, key=lambda rv: rv[0])

	breakpoints = [0.0]
	for i in range(len(sorted_) - 1):
		r0i, r1i = sorted_[i][0],   sorted_[i][1]
		r0j, r1j = sorted_[i+1][0], sorted_[i+1][1]
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
	print(f"  preference regions (w = weight on obj_0, 1-w on obj_1):")
	print(f"  {'obj_0':>10}  {'obj_1':>8}  {'w_left':>8}  {'w_right':>8}  {'length':>8}")
	for rv, wLeft, wRight in regions:
		print(f"  {rv[0]:>10.1f}  {rv[1]:>8.2f}  {wLeft:>8.4f}  {wRight:>8.4f}  {wRight - wLeft:>8.4f}")
	print()


# returns the total number of discrete states implied by the observation space
# supports Discrete and integer-valued Box spaces of any shape
def getStateSize(env):
	obsSpace = env.observation_space
	if isinstance(obsSpace, gymnasium.spaces.Discrete):
		return int(obsSpace.n)
	lo = np.floor(obsSpace.low).astype(int)
	hi = np.floor(obsSpace.high).astype(int)
	return int(np.prod(hi - lo + 1))


# maps an observation to a flat integer state index (mixed-radix encoding for Box spaces)
def obsToStateIdx(obs, env):
	obsSpace = env.observation_space
	if isinstance(obsSpace, gymnasium.spaces.Discrete):
		return int(obs)
	lo = np.floor(obsSpace.low).astype(int)
	hi = np.floor(obsSpace.high).astype(int)
	dims = hi - lo + 1
	obsInt = np.atleast_1d(np.asarray(obs, dtype=float)).astype(int)
	idx = 0
	for i in range(len(dims)):
		idx = idx * int(dims[i]) + int(obsInt[i]) - int(lo[i])
	return idx


# runs one episode using the provided policy callable; returns (trajectory, totalReward)
# policy(obs, env) -> action int
def runEpisode(env, policy):
	obs, info = env.reset()
	trajectory = []
	totalReward = np.zeros(env.unwrapped.reward_space.shape[0])
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


# returns the CCS region whose weight interval contains prefWeight (2D only)
def findRegion(prefWeight, regions):
	w = float(prefWeight[0])
	for region in regions:
		if region['wLeft'] <= w <= region['wRight']:
			return region
	return regions[-1]


# replays a stored trajectory graphically by stepping through a fresh render env
def renderTrajectory(traj, label, envName, frameDelay=0.4):
	renderEnv = mo_gym.make(envName, render_mode="human")
	renderEnv.reset()
	print(f"  replaying: {label}  ({len(traj)} steps)")
	renderEnv.render()
	time.sleep(frameDelay)
	for _, action, _ in traj:
		renderEnv.step(action)
		renderEnv.render()
		time.sleep(frameDelay)
	time.sleep(5.0)
	renderEnv.close()


# returns a policy callable that samples from the rho-adjusted distribution π*(a|s)^rho
def makeDemoPolicy(policy, rho):
	def demoPolicy(obs, env):
		s = tuple(obs)
		if s not in policy:
			return env.action_space.sample()
		probs = policy[s] ** rho
		probs /= probs.sum()
		return np.random.choice(len(probs), p=probs)
	return demoPolicy

