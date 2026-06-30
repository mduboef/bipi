
import os, sys, pickle
import numpy as np
from collections import defaultdict
from helpers import makeEnv, printEnvInfo, runEpisode, renderTrajectory
from algs.dwpi import makeBoltzmannDemoPolicy, _softV
from config import (TRAJ_PER_USER, NUM_USERS, POLICY_BETA, DEMO_BETA,
	CCS_EPISODES, CCS_GAMMA)


# no training/testing split, all this is test set data

# trains a tabular Q-table for a single preference weight w via epsilon-greedy Q-learning
# Creates same Boltzmann-rationality demonstrator model assumed by BIPI/DWPI
# scalarizes the vector reward by w, so this works for any number of objectives
# returns dict: state_tuple -> np.array(nActions,)
def trainQTable(env, w, nEpisodes, gamma, alpha=0.2, logEvery=2000):
      nActions = env.action_space.n
      qTable = defaultdict(lambda: np.zeros(nActions))
      epsStart, epsEnd = 0.8, 1e-4
      deltaWindow = []

      for ep in range(nEpisodes):
              eps = max(epsStart * (1.0 - ep / nEpisodes), epsEnd)
              obs, _ = env.reset()
              s = tuple(obs)
              done = False
              epMaxDelta = 0.0

              while not done:
                      if np.random.random() < eps:
                              a = env.action_space.sample()
                      else:
                              a = int(np.argmax(qTable[s]))

                      nextObs, r, terminated, truncated, _ = env.step(a)
                      ns = tuple(nextObs)
                      done = terminated or truncated

                      scalarR = float(np.dot(r, w))
                      target = scalarR if done else scalarR + gamma * _softV(qTable[ns], DEMO_BETA)
                      update = alpha * (target - qTable[s][a])
                      qTable[s][a] += update
                      epMaxDelta = max(epMaxDelta, abs(update))
                      s = ns

              deltaWindow.append(epMaxDelta)
              # print smoothed max Q-delta so convergence is visible as the curve flattens
              if logEvery and (ep + 1) % logEvery == 0:
                      print(f"    ep {ep+1:>6}/{nEpisodes}  eps {eps:.3f}  avgMaxDelta {np.mean(deltaWindow):.5f}", flush=True)
                      deltaWindow = []

      return dict(qTable)


# saves demo/rollouts to disk with their associated w
# overwrites the file each call so partial progress is preserved if training is interrupted
def saveTestData(env, envName, testWs, testPolicies, testDemos):
	dataDir = os.path.join(os.path.dirname(__file__), "testingData")
	os.makedirs(dataDir, exist_ok=True)
	outPath = os.path.join(dataDir, f"{envName}_testData.pkl")

	data = {
		'envName': envName,
		'demoBeta': DEMO_BETA,
		'ws': testWs,
		'demos': testDemos,
	}

	with open(outPath, 'wb') as f:
		pickle.dump(data, f)

	print(f"\tSaved {len(testDemos[0])*len(testWs)} demos from {len(testWs)} synthetic demonstrators to {outPath}")


# writes a single object to disk atomically (tmp file + os.replace) so a crash mid-write can't corrupt it
def writePickleAtomic(obj, path):
	tmp = path + ".tmp"
	with open(tmp, "wb") as f:
		pickle.dump(obj, f)
	os.replace(tmp, path)


# trains one synthetic demonstrator end to end inside its own process
# creates a private env, seeds its rng, trains a Q-table, rolls out demos, writes a per-user shard
# must stay a top-level function so ProcessPoolExecutor can pickle it by name
# returns the userIdx so the driver can track progress with minimal IPC
def trainOneUser(taskArgs):
	envName, userIdx, seed, shardDir = taskArgs

	# seed every source of randomness so workers don't share a fork-inherited rng stream
	np.random.seed(seed)
	env = makeEnv(envName)
	env.action_space.seed(seed)

	numObjectives = env.unwrapped.reward_space.shape[0]
	rng = np.random.default_rng(seed)
	w = rng.dirichlet(np.ones(numObjectives))

	# logEvery=0 keeps the worker silent so output from many processes doesn't interleave
	qTable = trainQTable(env, w, CCS_EPISODES[envName], CCS_GAMMA[envName], logEvery=0)
	policy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)

	userDemos = []
	for j in range(TRAJ_PER_USER):
		demo, _ = runEpisode(env, policy)
		userDemos.append(demo)
	env.close()

	shardPath = os.path.join(shardDir, f"{envName}_user_{userIdx:05d}.pkl")
	writePickleAtomic({"idx": userIdx, "w": w, "demos": userDemos}, shardPath)
	return userIdx


# merges all per-user shards into the single testData.pkl that main.py consumes
# loads shards in index order so ws[i] stays paired with demos[i]
def mergeShards(envName, shardDir, outPath, numUsers):
	shards = []
	for idx in range(numUsers):
		shardPath = os.path.join(shardDir, f"{envName}_user_{idx:05d}.pkl")
		if os.path.exists(shardPath):
			with open(shardPath, "rb") as f:
				shards.append(pickle.load(f))

	shards.sort(key=lambda d: d["idx"])
	testWs = [s["w"] for s in shards]
	testDemos = [s["demos"] for s in shards]

	data = {
		"envName": envName,
		"demoBeta": DEMO_BETA,
		"ws": testWs,
		"demos": testDemos,
	}
	writePickleAtomic(data, outPath)
	print(f"\tMerged {len(testWs)} users -> {outPath}")
	return data


# parallel demonstrator generation across cpu cores
# each user is i.i.d. so we fan them out with ProcessPoolExecutor and resume from existing shards
# numWorkers defaults to all cores; outDir should point at persistent storage (eg a Drive folder) on colab
# userStart/userEnd bound the index slice this run trains, so multiple colab sessions can split the
# work over one shared shard folder by claiming disjoint ranges (eg 0-2500 and 2500-5000)
def testSetGenParallel(envName, numUsers=NUM_USERS, numWorkers=None, outDir=None, baseSeed=0,
		userStart=0, userEnd=None):
	from concurrent.futures import ProcessPoolExecutor, as_completed

	if numWorkers is None:
		numWorkers = os.cpu_count()
	if outDir is None:
		outDir = os.path.join(os.path.dirname(__file__), "testingData")
	if userEnd is None:
		userEnd = numUsers

	shardDir = os.path.join(outDir, f"{envName}_shards")
	os.makedirs(shardDir, exist_ok=True)
	outPath = os.path.join(outDir, f"{envName}_testData.pkl")

	# resume: only train users in this run's slice whose shard is missing
	todo = []
	for idx in range(userStart, userEnd):
		shardPath = os.path.join(shardDir, f"{envName}_user_{idx:05d}.pkl")
		if not os.path.exists(shardPath):
			todo.append(idx)

	print(f"{numUsers - len(todo)}/{numUsers} already done; training {len(todo)} more on {numWorkers} workers", flush=True)
	if not todo:
		return mergeShards(envName, shardDir, outPath, numUsers)

	# seed per user index so reruns and resumes stay reproducible
	tasks = [(envName, idx, baseSeed + idx, shardDir) for idx in todo]

	completed = 0
	with ProcessPoolExecutor(max_workers=numWorkers) as ex:
		futures = [ex.submit(trainOneUser, t) for t in tasks]
		for fut in as_completed(futures):
			idx = fut.result()
			completed += 1
			print(f"user {idx} done ({completed}/{len(todo)})", flush=True)

	return mergeShards(envName, shardDir, outPath, numUsers)


# trains synthetic demostrator policies with random ws
# saves the w, rollouts and policy to disk
def testSetGen():
	
	# usage: python3 testSetGen.py <envName>

	envName = sys.argv[1]

	env = makeEnv(envName)
	printEnvInfo(env, envName)

	# number of objectives is the dimensionality of the vector reward
	numObjectives = env.unwrapped.reward_space.shape[0]

	testWs = []
	testPolicies = []
	testDemos = []

	# randomly select NUM_USERS preference weights
	# TODO parallelize this so we are training multiple "users"/demonstrators at a time
	for i in range(NUM_USERS):

		# generate the randomized pref vector
		rng = np.random.default_rng()
		w = rng.dirichlet(np.ones(numObjectives))

		testWs.append(w)

		print(f"Training Demonstrator {i+1}...")
		print(f"\tw = {w}")		# TODO print this in a more organized and presentable way

		# train a RL policy specific to the exact preference weight
		# ? should I train till convergence instead of a fixed num of eps
		qTable = trainQTable(env, w, CCS_EPISODES[envName], CCS_GAMMA[envName])		# NOTE demonstrators trained with same # of episodes as used to train CCS policies, is there a more principled stopping criteria
		policy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)
		testPolicies.append(policy)

		# for each user generate TRAJ_PER_USER demos
		userDemos = []
		for j in range(TRAJ_PER_USER):

			# rolls out policy for one episode
			# returns the trajectory: list of (obs, action, reward) tuples
			demo, _ = runEpisode(env, policy)
			userDemos.append(demo)

		testDemos.append(userDemos)

	# save all these demos to disk with their associated label (pref weight)
	saveTestData(env, envName, testWs, testPolicies, testDemos)

	# render trajectories if we have the -render flag
	renderBool = ("-render" in sys.argv)
	if renderBool:
		for i in range(NUM_USERS):
			print(f"Displaying demos by user {i}")
			print(f"w = {testWs[i]}")		# TODO print this in a more organized and presentable way
			for j in range(TRAJ_PER_USER):
				print(f"\tTrajectory {j+1}/{TRAJ_PER_USER}...")
				renderTrajectory(testDemos[i][j], testWs[i], envName, frameDelay=0.4)

	return

# only run the serial path when executed directly as a script, so the notebook can import the
# parallel helpers without kicking off a 5000-user serial run
if __name__ == "__main__":
	testSetGen()





# LATER IN MAIN.PY

# run all methods on the testing data saved to disk: bipi & different versions of dwpi
# select a single inferred preference weight for each method
# 	for dwpi methods this is just the output of the network
# 	for bipi this means using different approaches to select a single discrete w given the bipi posterior
# select policy from ccs to corresponds to inferred preference
# compare results
# 	in terms of expected reward for selected CCS policy under ground truth preference (label)
# 		how much worse was selected CCS policy than the best/optimal one?
# 	in terms of pref inference accuracy & distance
# 		how close were they to the correct w?
# 		how often did they select the best/optimal ccs policy?