import os
import sys
import numpy as np

from config import (POLICY_BETA, DEMO_BETA, NUM_USERS, TRAJ_PER_USER,
	DWPI_GRANULARITY, DWPI_SF_GAMMA, DWPI_HIDDEN_DIM)
from helpers import runEpisode, findRegion
from algs.ccs import loadCCS
from algs.bipi import (runBIPI, selectMapPolicy, selectMeanWeightPolicy,
	selectEUPolicy, selectCVaRPolicy)
from algs.dwpi import (ENCODINGS, getWeightVecs, getGridDims, lookupQTable,
	makeBoltzmannDemoPolicy, encodeDemos, loadDWMOTQ, loadModels, inferWeight)

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


def _evaluateUser(env, regions, qTables, models, weightVecs, nRows, nCols, rho, user):
	nObj       = len(regions[0]['returnVec'])
	prefWeight = np.random.dirichlet(np.ones(nObj))

	qTable     = lookupQTable(qTables, prefWeight, weightVecs)
	demoPolicy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)
	demos      = [runEpisode(env, demoPolicy)[0] for _ in range(TRAJ_PER_USER)]

	trueRegion    = findRegion(prefWeight, regions)
	trueRegionIdx = next(i for i, r in enumerate(regions) if r is trueRegion)
	optimalEU     = float(np.dot(prefWeight, regions[trueRegionIdx]['returnVec']))

	posterior = runBIPI(regions, demos, rho)
	assigned = {
		'bipi_map':  selectMapPolicy(posterior),
		'bipi_mean': selectMeanWeightPolicy(posterior, regions),
		'bipi_eu':   selectEUPolicy(posterior, regions),
		'bipi_cvar': selectCVaRPolicy(posterior, regions, 0.05),
	}

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

	print(f"  user {user:>3}: region {trueRegionIdx}  "
	      + '  '.join(f"{k}: {'✓' if v['correct'] else '✗'}"
	                  for k, v in userResult.items() if isinstance(v, dict)))
	return userResult


# loads all artifacts and runs the head-to-head comparison between BIPI and DWPI
def runCompare(env, ccsDir, saveDir):
	ccs = loadCCS(ccsDir)
	regions = ccs['regions']

	nObj         = len(regions[0]['returnVec'])
	nRows, nCols = getGridDims(env)
	stateSize    = nRows * nCols
	weightVecs   = getWeightVecs(DWPI_GRANULARITY)
	rho          = DEMO_BETA / POLICY_BETA

	dwmotqPath = os.path.join(saveDir, 'dwmotq.pkl')
	modelPaths = {enc: os.path.join(saveDir, f'model_{enc}.pt') for enc in ENCODINGS}
	missing = ([] if os.path.exists(dwmotqPath) else ['dwmotq.pkl']) + \
	          [f'model_{enc}.pt' for enc, p in modelPaths.items() if not os.path.exists(p)]
	if missing:
		print("Error: missing cached files — run 'dwpi' mode first to generate them:")
		for m in missing: print(f"  {m}")
		sys.exit(1)

	print("Loading DWMOTQ agent and inference models ...")
	qTables = loadDWMOTQ(saveDir)
	models  = loadModels({'return': nObj, 'stateFreq': stateSize, 'sf': stateSize},
	                     nObj, DWPI_HIDDEN_DIM, saveDir)

	results = []
	for user in range(NUM_USERS):
		result = _evaluateUser(env, regions, qTables, models, weightVecs, nRows, nCols, rho, user)
		results.append(result)

	printCompareResults(results)
