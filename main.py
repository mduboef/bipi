# Bayesian Multi-Objective Preference Inference from Demonstration (BMOPID)
	# also sometimes refered to as Bayesian Inferse Preference Inference (BIPI) since I dont know what to call it

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

import os
import sys
import numpy as np
from config import (CCS_EPISODES, CCS_GAMMA, POLICY_BETA, DEMO_BETA, TRAJ_PER_USER, NUM_USERS,
	DWPI_GRANULARITY, DWPI_N_EPISODES, DWPI_NDEMOS_TRAIN, DWPI_AUGMENT,
	DWPI_SF_GAMMA, DWPI_HIDDEN_DIM, DWPI_EPOCHS, DWPI_LR, DWPI_NDEMOS_INFER)
from helpers import makeEnv, printEnvInfo, runEpisode, findRegion, renderTrajectory, getStateSize
from algs.ccs import buildCCS, printCCS, saveCCS, loadCCS, trainSoftmaxPolicies, saveSoftmaxPolicies, loadSoftmaxPolicies
from algs.bipi import runBIPI, selectMapPolicy, selectMeanWeightPolicy, selectEUPolicy, selectCVaRPolicy, printBIPIResults, printBIPIAccuracy
from algs.dwpi import (ENCODINGS, getWeightVecs,
	trainDWMOTQ, saveDWMOTQ, loadDWMOTQ, lookupQTable,
	makeBoltzmannDemoPolicy, makeGreedyPolicy,
	buildDWPIDataset, saveDataset, loadDataset,
	trainDWPIModel, saveModels, loadModels,
	generateTestEncodings, inferWeight, printDWPIResults)
from compare import runCompare


# testGen.py
	# randomly select NUM_USERS preference weights
	# train a RL policy specific to each users exact weight
	# for each user generate TRAJ_PER_USER demos
	# no training/testing split, all this is test set data
	# save all these demos to disk with their associated label (pref weight)

# main.py
	# read all testing data from the disk after running testGen.py
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



def main():

	# stores names of environments to run experiments on
	# TODO adapt existing code to work with resource-gathering-v0 environment
		# requires changing code to work with 3+ objectives
	envNames = ["deep-sea-treasure-v0", "fishwood-v0", "resource-gathering-v0"]
	methodNames = ["ccs", "bipi", "dwpi", "compare"]


	if sys.argv[1] not in envNames:
		print(f"Usage: python main.py <envName> <methodName>")
		print(f"  where envName is one of: {envNames}")
		sys.exit(1)
	else: envName = sys.argv[1]

	if envName not in CCS_EPISODES or envName not in CCS_GAMMA:
		print(f"Error: no CCS config for '{envName}'.")
		print(f"  Add entries to CCS_EPISODES and CCS_GAMMA in config.py and re-run.")
		sys.exit(1)
	nEpisodes = CCS_EPISODES[envName]
	ccsGamma  = CCS_GAMMA[envName]

	if sys.argv[2] not in methodNames:
		print(f"Usage: python main.py <envName> <methodName>")
		print(f"  where methodName is one of: {methodNames}")
		sys.exit(1)
	else: methodName = sys.argv[2]

	# flag will graphically render synthetic demos and rollout of CCS policy that would be best for user
	renderBool = ("-render" in sys.argv)
	
	# create env object
	env = makeEnv(envName)
	printEnvInfo(env, envName)

	# number of objectives is the dimensionality of the vector reward
	# TODO update all functionality to work with variable number of objectives, numObjectives
	numObjectives = env.unwrapped.reward_space.shape[0]



	# CCS METHOD:	calulate ccs policies (no inference)
	if methodName == "ccs":
		# NOTE the deep-sea-treasure-v0 and resource-gathering-v0 envs come with a precomptued pareto front, fishwood-v0 doesn't
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		ccsPath = os.path.join(saveDir, 'ccs.pkl')
		if os.path.exists(ccsPath):
			print("Loading existing CCS ...")
			ccs = loadCCS(saveDir)['regions']
		else:
			print(f"Calculating CCS (beta = {POLICY_BETA}, gamma = {ccsGamma}) ...")
			ccs = buildCCS(env, nEpisodes=nEpisodes, gamma=ccsGamma)
			printCCS(ccs)
			saveCCS(ccs, saveDir)

		print(f"Training softmax policies at region centroids (beta = {POLICY_BETA}, gamma = {ccsGamma}) ...")
		softmaxPolicies = trainSoftmaxPolicies(env, ccs, POLICY_BETA, nEpisodes=nEpisodes, gamma=ccsGamma)
		saveSoftmaxPolicies(softmaxPolicies, saveDir)

		# TODO train softmax policies at 101 discrete and evenly spaced ws, using the same granularity as DWPI

	# BIPI METHOD:	use new bayesean inference approach on saved demo data
	elif methodName == "bipi":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		ccs = loadCCS(ccsDir)
		regions = ccs['regions']

		softmaxPath = os.path.join(ccsDir, 'softmaxPolicies.pkl')
		if not os.path.exists(softmaxPath):
			print(f"Error: softmax policies not found at {softmaxPath}. Run 'ccs' mode first.")
			sys.exit(1)
		print("Loading softmax policies ...")
		softmaxPolicies = loadSoftmaxPolicies(ccsDir)
		for sp in softmaxPolicies:
			regions[sp['regionIdx']]['policy'] = sp['policy']

		dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
		if not os.path.exists(dwmotqPath):
			print(f"Error: DWMOTQ not found at {dwmotqPath}. Run 'dwpi' mode first to train and cache it.")
			sys.exit(1)
		print("Loading DWMOTQ agent ...")
		qTables    = loadDWMOTQ(saveDir)
		weightVecs = getWeightVecs(DWPI_GRANULARITY)

		nObj = len(regions[0]['returnVec'])
		rho  = DEMO_BETA / POLICY_BETA
		bipiResults = []

		for user in range(NUM_USERS):
			prefWeight = np.random.dirichlet(np.ones(nObj))
			print(f"\nUser {user}: true preference weight = {np.round(prefWeight, 4)}")

			# generate demos: Boltzmann-rational at DEMO_BETA using the DWMOTQ Q-table for the nearest discretized weight
			qTable     = lookupQTable(qTables, prefWeight, weightVecs)
			demoPolicy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)
			demos = [runEpisode(env, demoPolicy)[0] for _ in range(TRAJ_PER_USER)]
			print(f"  generated {TRAJ_PER_USER} demo(s), {sum(len(d) for d in demos)} total steps")

			if renderBool:
				for i, demo in enumerate(demos):
					renderTrajectory(demo, f"demo {i}  (DEMO_BETA={DEMO_BETA})", envName)
				optimalTraj, _ = runEpisode(env, makeGreedyPolicy(qTable))
				renderTrajectory(optimalTraj, f"greedy DWMOTQ rollout", envName)

			trueRegion    = findRegion(prefWeight, regions)
			trueRegionIdx = next(i for i, r in enumerate(regions) if r is trueRegion)

			posterior = runBIPI(regions, demos, rho)

			mapIdx  = selectMapPolicy(posterior)
			meanIdx = selectMeanWeightPolicy(posterior, regions)
			euIdx   = selectEUPolicy(posterior, regions)
			cvarIdx = selectCVaRPolicy(posterior, regions, 0.05)

			printBIPIResults(posterior, regions, trueRegionIdx, prefWeight, mapIdx, meanIdx, euIdx, cvarIdx)
			bipiResults.append({'trueRegionIdx': trueRegionIdx, 'map': mapIdx, 'mean': meanIdx, 'eu': euIdx, 'cvar': cvarIdx})

		printBIPIAccuracy(bipiResults)


	# DWPI METHOD:	use new baseline inference approach on saved demo data
	elif methodName == "dwpi":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		ccs     = loadCCS(ccsDir)
		regions = ccs['regions']
		nObj      = len(regions[0]['returnVec'])
		stateSize = getStateSize(env)

		# step 1:	discretize the pref space, evently spaced
		weightVecs = getWeightVecs(DWPI_GRANULARITY)

		# step 2:	train one Q-table per pref weight, w, using DWMOTQ (tabular Q learning)
		dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
		if os.path.exists(dwmotqPath):			# load the q tables from disk if they already exist
			print("Loading Q tables that exist on disk in dwmotq.pkl...")
			qTables = loadDWMOTQ(saveDir)		# NOTE delete dwmotq.pkl if you want new Q tables
		else:
			print(f"Training {len(weightVecs)} Q tables ({DWPI_N_EPISODES} episodes each) ...")
			qTables = trainDWMOTQ(env, DWPI_GRANULARITY, DWPI_N_EPISODES)
			saveDWMOTQ(qTables, saveDir)

		# step 3:	build set of training data, turning each Q-table into actual trajectories
		datasetPath = os.path.join(saveDir, 'dataset.pkl')
		if os.path.exists(datasetPath):			# load the training data from disk if they already exist
			print("Loading train data that exist on disk in dataset.pkl...")
			Xret, Xfreq, Xsf, Y = loadDataset(saveDir)
		else:
			nExamples = len(weightVecs) * DWPI_AUGMENT
			print(f"Building dataset ({len(weightVecs)} weights × {DWPI_AUGMENT} augmentations "
			      f"× {DWPI_NDEMOS_TRAIN} demos = {nExamples} examples) ...")
			Xret, Xfreq, Xsf, Y = buildDWPIDataset(
				env, qTables, DWPI_GRANULARITY, DWPI_NDEMOS_TRAIN, DWPI_AUGMENT, DWPI_SF_GAMMA)
			saveDataset((Xret, Xfreq, Xsf, Y), saveDir)

		# step 4:	train or load one FNN per encoding
		inputDims   = {'return': nObj, 'stateFreq': stateSize, 'sf': stateSize}
		modelPaths  = {enc: os.path.join(saveDir, f'model_{enc}.pt') for enc in ENCODINGS}
		XbyEnc      = {'return': Xret, 'stateFreq': Xfreq, 'sf': Xsf}

		if all(os.path.exists(p) for p in modelPaths.values()):
			print("Loading trained models ...")
			models = loadModels(inputDims, nObj, DWPI_HIDDEN_DIM, saveDir)
		else:
			print(f"Training inference models ({DWPI_EPOCHS} epochs each) ...")
			models = {}
			for enc in ENCODINGS:
				print(f"  [{enc}]")
				models[enc] = trainDWPIModel(
					XbyEnc[enc], Y, nObj, DWPI_HIDDEN_DIM, DWPI_EPOCHS, DWPI_LR)
			saveModels(models, saveDir)

		# step 5:	evaluate on simulated users
		# sample test weights from the discretized set (same distribution as training)
		userResults = []
		for user in range(NUM_USERS):
			trueW       = weightVecs[np.random.randint(len(weightVecs))]
			trueRegion  = findRegion(trueW, regions)
			trueRegionIdx = next(i for i, r in enumerate(regions) if r is trueRegion)
			print(f"\nUser {user}: w = {np.round(trueW, 4)}  (region {trueRegionIdx})")

			# generate DWPI_NDEMOS_INFER demos using the DWMOTQ agent for this weight
			encFeats = generateTestEncodings(
				env, qTables, trueW, DWPI_NDEMOS_INFER, DWPI_SF_GAMMA, weightVecs)

			inferred          = {}
			inferredRegionIdx = {}
			for enc in ENCODINGS:
				infW      = inferWeight(models[enc], encFeats[enc])
				infRegion = findRegion(infW, regions)
				infIdx    = next(i for i, r in enumerate(regions) if r is infRegion)
				inferred[enc]          = infW
				inferredRegionIdx[enc] = infIdx
				ok = 'correct' if infIdx == trueRegionIdx else 'wrong'
				print(f"  {enc:<10}: ŵ = {np.round(infW, 4)}  → region {infIdx}  ({ok})")

			userResults.append({
				'user': user, 'trueWeight': trueW, 'trueRegionIdx': trueRegionIdx,
				'inferred': inferred, 'inferredRegionIdx': inferredRegionIdx,
			})

		printDWPIResults(userResults, regions)



	# COMPARE METHOD: lunches comapritive analysis of bipi and dwpi, using saved dwpi model and ccs policies on disk
	elif methodName == "compare":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		runCompare(env, ccsDir, saveDir)


	env.close()

main()