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
from config import (POLICY_BETA, DEMO_BETA, TRAJ_PER_USER, NUM_USERS,
	DWPI_GRANULARITY, DWPI_N_EPISODES, DWPI_NDEMOS_TRAIN, DWPI_AUGMENT,
	DWPI_SF_GAMMA, DWPI_HIDDEN_DIM, DWPI_EPOCHS, DWPI_LR, DWPI_NDEMOS_INFER)
from helpers import makeEnv, printEnvInfo, runEpisode, findRegion, renderTrajectory
from algs.ccs import buildCCS, printCCS, saveCCS, loadCCS
from algs.bipi import runBIPI, selectMapPolicy, selectMeanWeightPolicy, selectEUPolicy, selectCVaRPolicy, printBIPIResults, printBIPIAccuracy
from algs.dwpi import (ENCODINGS, getWeightVecs, getGridDims, findNearestWeight,
	trainDWMOTQ, saveDWMOTQ, loadDWMOTQ, lookupQTable,
	makeBoltzmannDemoPolicy, makeGreedyPolicy, encodeDemos,
	buildDWPIDataset, saveDataset, loadDataset,
	trainDWPIModel, saveModels, loadModels,
	generateTestEncodings, inferWeight, printDWPIResults)




_COMPARE_METHODS = [
	('bipi:MAP',        'bipi_map'),
	('bipi:mean',       'bipi_mean'),
	('bipi:EU',         'bipi_eu'),
	('bipi:CVaR(0.05)', 'bipi_cvar'),
	('dwpi:return',     'dwpi_return'),
	('dwpi:stateFreq',  'dwpi_stateFreq'),
	('dwpi:sf',         'dwpi_sf'),
]

def printCompareResults(results):
	nUsers = len(results)
	print(f"\n  {'method':<18}  {'accuracy':>15}  {'mean util gap':>14}")
	print(f"  {'-'*18}  {'-'*15}  {'-'*14}")
	for label, key in _COMPARE_METHODS:
		nCorrect = sum(1 for r in results if r[key]['correct'])
		meanGap  = np.mean([r[key]['eu'] - r['optimalEU'] for r in results])
		print(f"  {label:<18}  {nCorrect:>3}/{nUsers}  ({nCorrect/nUsers*100:>5.1f}%)  {meanGap:>+.4f}")


def main():

	# stores names of environments to run experiments on
	envNames = ["deep-sea-treasure-v0"]
	methodNames = ["ccs", "bipi", "dwpi", "compare"]


	if sys.argv[1] not in envNames:
		print(f"Usage: python main.py <envName> <methodName>")
		print(f"  where envName is one of: {envNames}")
		sys.exit(1)
	else: envName = sys.argv[1]

	if sys.argv[2] not in methodNames:
		print(f"Usage: python main.py <envName> <methodName>")
		print(f"  where methodName is one of: {methodNames}")
		sys.exit(1)
	else: methodName = sys.argv[2]

	# flag will graphically render synthetic demos and rollout of CCS policy that would be best for user
	renderBool = ("-render" in sys.argv)
	

	env = makeEnv()
	printEnvInfo(env, envName)


	if methodName == "ccs":
		print(f"Calculating CCS (beta = {POLICY_BETA}) ...")
		ccs = buildCCS(env, nEpisodes=100000)
		printCCS(ccs)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveCCS(ccs, saveDir, POLICY_BETA)


	elif methodName == "bipi":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		ccs = loadCCS(ccsDir)
		if ccs['policyBeta'] != POLICY_BETA:
			print(f"Warning: CCS was computed with policyBeta={ccs['policyBeta']} but config has POLICY_BETA={POLICY_BETA}")
		regions = ccs['regions']

		dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
		if not os.path.exists(dwmotqPath):
			print(f"Error: DWMOTQ not found at {dwmotqPath}. Run 'dwpi' mode first to train and cache it.")
			sys.exit(1)
		print("Loading DWMOTQ agent ...")
		qTables    = loadDWMOTQ(saveDir)
		weightVecs = getWeightVecs(DWPI_GRANULARITY)

		nObj = len(regions[0]['returnVec'])
		rho  = DEMO_BETA / ccs['policyBeta']  # used by BIPI inference likelihood model only
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


	elif methodName == "dwpi":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		ccs     = loadCCS(ccsDir)
		regions = ccs['regions']
		nObj    = len(regions[0]['returnVec'])
		nRows, nCols = getGridDims(env)
		stateSize    = nRows * nCols
		weightVecs   = getWeightVecs(DWPI_GRANULARITY)

		# phase 1: DWMOTQ — train or load one Q-table per discretized weight
		dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
		if os.path.exists(dwmotqPath):
			print("Loading DWMOTQ agent ...")
			qTables = loadDWMOTQ(saveDir)
		else:
			print(f"Training DWMOTQ ({len(weightVecs)} weights × {DWPI_N_EPISODES} episodes each) ...")
			qTables = trainDWMOTQ(env, DWPI_GRANULARITY, DWPI_N_EPISODES)
			saveDWMOTQ(qTables, saveDir)

		# phase 2: synthetic training dataset — build or load
		datasetPath = os.path.join(saveDir, 'dataset.pkl')
		if os.path.exists(datasetPath):
			print("Loading dataset ...")
			Xret, Xfreq, Xsf, Y = loadDataset(saveDir)
		else:
			nExamples = len(weightVecs) * DWPI_AUGMENT
			print(f"Building dataset ({len(weightVecs)} weights × {DWPI_AUGMENT} augmentations "
			      f"× {DWPI_NDEMOS_TRAIN} demos = {nExamples} examples) ...")
			Xret, Xfreq, Xsf, Y = buildDWPIDataset(
				env, qTables, DWPI_GRANULARITY, DWPI_NDEMOS_TRAIN, DWPI_AUGMENT, DWPI_SF_GAMMA)
			saveDataset((Xret, Xfreq, Xsf, Y), saveDir)

		# phase 3: train or load one FNN per encoding
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

		# phase 4: evaluate on simulated users
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

	elif methodName == "compare":
		ccsDir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		saveDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dwpiResults', envName)
		ccs = loadCCS(ccsDir)
		if ccs['policyBeta'] != POLICY_BETA:
			print(f"Warning: CCS was computed with policyBeta={ccs['policyBeta']} but config has POLICY_BETA={POLICY_BETA}")
		regions = ccs['regions']

		# require both DWMOTQ and trained inference models — refuse to run if either is missing
		dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
		modelPaths = {enc: os.path.join(saveDir, f'model_{enc}.pt') for enc in ENCODINGS}
		missing = ([] if os.path.exists(dwmotqPath) else ['dwmotq.pkl']) + \
		          [f'model_{enc}.pt' for enc, p in modelPaths.items() if not os.path.exists(p)]
		if missing:
			print("Error: missing cached files — run 'dwpi' mode first to generate them:")
			for m in missing: print(f"  {m}")
			sys.exit(1)

		print("Loading DWMOTQ agent and inference models ...")
		qTables      = loadDWMOTQ(saveDir)
		weightVecs   = getWeightVecs(DWPI_GRANULARITY)
		nRows, nCols = getGridDims(env)
		stateSize    = nRows * nCols
		nObj         = len(regions[0]['returnVec'])
		models       = loadModels({'return': nObj, 'stateFreq': stateSize, 'sf': stateSize},
		                          nObj, DWPI_HIDDEN_DIM, saveDir)
		rho          = DEMO_BETA / ccs['policyBeta']

		results = []
		for user in range(NUM_USERS):
			prefWeight = np.random.dirichlet(np.ones(nObj))

			# generate shared demos used by all methods
			qTable     = lookupQTable(qTables, prefWeight, weightVecs)
			demoPolicy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)
			demos      = [runEpisode(env, demoPolicy)[0] for _ in range(TRAJ_PER_USER)]

			trueRegion    = findRegion(prefWeight, regions)
			trueRegionIdx = next(i for i, r in enumerate(regions) if r is trueRegion)
			optimalEU     = float(np.dot(prefWeight, regions[trueRegionIdx]['returnVec']))

			# BIPI: Bayesian posterior → four policy selection strategies
			posterior = runBIPI(regions, demos, rho)
			assigned = {
				'bipi_map':  selectMapPolicy(posterior),
				'bipi_mean': selectMeanWeightPolicy(posterior, regions),
				'bipi_eu':   selectEUPolicy(posterior, regions),
				'bipi_cvar': selectCVaRPolicy(posterior, regions, 0.05),
			}

			# DWPI: encode same demos → infer weight → find region
			for enc in ENCODINGS:
				feat      = encodeDemos(demos, enc, nRows, nCols, DWPI_SF_GAMMA)
				infW      = inferWeight(models[enc], feat)
				infRegion = findRegion(infW, regions)
				assigned['dwpi_' + enc] = next(i for i, r in enumerate(regions) if r is infRegion)

			userResult = {'user': user, 'prefWeight': prefWeight,
			              'trueRegionIdx': trueRegionIdx, 'optimalEU': optimalEU}
			for key, idx in assigned.items():
				userResult[key] = {'regionIdx': idx,
				                   'eu':      float(np.dot(prefWeight, regions[idx]['returnVec'])),
				                   'correct': idx == trueRegionIdx}
			results.append(userResult)
			print(f"  user {user:>3}: region {trueRegionIdx}  "
			      + '  '.join(f"{k}: {'✓' if v['correct'] else '✗'}"
			                  for k, v in userResult.items() if isinstance(v, dict)))

		printCompareResults(results)


	env.close()

main()