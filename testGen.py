
import os, sys
import numpy as np
from collections import defaultdict
from helpers import makeEnv, printEnvInfo, runEpisode
from algs.dwpi import makeBoltzmannDemoPolicy, _softV
from config import (TRAJ_PER_USER, NUM_USERS, POLICY_BETA, DEMO_BETA,
	CCS_EPISODES, CCS_GAMMA)


# no training/testing split, all this is test set data

# trains a tabular Q-table for a single preference weight w via epsilon-greedy Q-learning
# Creates same Boltzmann-rationality demonstrator model assumed by BIPI/DWPI
# scalarizes the vector reward by w, so this works for any number of objectives
# returns dict: state_tuple -> np.array(nActions,)
def trainQTable(env, w, nEpisodes, gamma, alpha=0.2):
	nActions = env.action_space.n
	qTable = defaultdict(lambda: np.zeros(nActions))
	epsStart, epsEnd = 0.8, 1e-4

	for ep in range(nEpisodes):
		eps = max(epsStart * (1.0 - ep / nEpisodes), epsEnd)
		obs, _ = env.reset()
		s = tuple(obs)
		done = False

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
			qTable[s][a] += alpha * (target - qTable[s][a])
			s = ns

	return dict(qTable)


# saves demo/rollouts to disk with their associated w
def saveTestData(env, testWs, testPolicies, testDemos):

	for userID in range(len(testWs)):
		# add an entry to the test data file for this user
			# includes w, beta and (state, action) data for all the user's rollouts/demos

		# OPTIONAL: save policy file
	return

# trains synthetic demostrator policies with random ws
# saves the w, trained, policy and rollouts to disk
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
	for i in range(NUM_USERS):

		
		# generate the randomized pref vector
		rng = np.random.default_rng()
		w = rng.dirichlet(np.ones(numObjectives-1))	# vec length is numObjectives-1

		testWs.append(w)

		# train a RL policy specific to the exact preference weight
		qTable = trainQTable(env, w, CCS_EPISODES[envName], CCS_GAMMA[envName])
		testPolicies.append(policy)

		# for each user generate TRAJ_PER_USER demos
		userDemos = []
		for j in range(TRAJ_PER_USER):

			# create boltzman rationality (DEMO_BETA) policy from Q table
			policy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)

			# rolls out policy for one episode
			# returns the trajectory: list of (obs, action, reward) tuples
			demo, _ = runEpisode(env, demoPolicy)
			userDemos.append(demo)

		testDemos.append(userDemos)

		# save all these demos to disk with their associated label (pref weight)
		saveTestData(testWs, testDemos)

	return

testSetGen():





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