import os
import sys
import numpy as np

from config import POLICY_BETA, DWPI_SF_GAMMA, DWPI_HIDDEN_DIM
from helpers import findRegion, getStateSize, obsToStateIdx, loadTestData
from algs.ccs import loadCCS, loadSoftmaxPolicies
from algs.bipi import (runBIPI, selectMapPolicy, selectMeanWeightPolicy,
	selectEUPolicy, selectCVaRPolicy)
from algs.dwpi import ENCODINGS, encodeDemos, loadModels, inferWeight

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


def _evaluateUser(env, regions, models, stateSize, rho, user, prefWeight, demos):
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

	obsToIdx = lambda obs: obsToStateIdx(obs, env)
	for enc in ENCODINGS:
		feat      = encodeDemos(demos, enc, stateSize, obsToIdx, DWPI_SF_GAMMA)
		infW      = inferWeight(models[enc], feat)
		infRegion = findRegion(infW, regions)
		assigned['dwpi_' + enc] = next(i for i, r in enumerate(regions) if r is infRegion)

	userResult = {'user': user, 'prefWeight': prefWeight,
	              'trueRegionIdx': trueRegionIdx, 'optimalEU': optimalEU}
	for key, idx in assigned.items():
		userResult[key] = {'regionIdx': idx,
		                   'eu':      float(np.dot(prefWeight, regions[idx]['returnVec'])),
		                   'correct': idx == trueRegionIdx}

	# print(f"  user {user:>3}: region {trueRegionIdx}  "
	#       + '  '.join(f"{k}: {'✓' if v['correct'] else '✗'}"
	#                   for k, v in userResult.items() if isinstance(v, dict)))
	return userResult


# loads all artifacts and runs the head-to-head comparison between BIPI and DWPI
# both methods are scored on the identical test demos saved to disk by testGen.py
def runCompare(env, envName, ccsDir, saveDir):
	ccs = loadCCS(ccsDir)
	regions = ccs['regions']

	softmaxPolicies = loadSoftmaxPolicies(ccsDir)
	for sp in softmaxPolicies:
		regions[sp['regionIdx']]['policy'] = sp['policy']

	nObj      = len(regions[0]['returnVec'])
	stateSize = getStateSize(env)

	modelPaths = {enc: os.path.join(saveDir, f'model_{enc}.pt') for enc in ENCODINGS}
	missing = [f'model_{enc}.pt' for enc, p in modelPaths.items() if not os.path.exists(p)]
	if missing:
		print("Error: missing cached DWPI models — run 'dwpi' mode first to generate them:")
		for m in missing: print(f"  {m}")
		sys.exit(1)

	print("Loading DWPI inference models ...")
	models  = loadModels({'return': nObj, 'stateFreq': stateSize, 'sf': stateSize},
	                     nObj, DWPI_HIDDEN_DIM, saveDir)

	# load the shared test set and score both methods on the same demonstrators
	testData = loadTestData(envName)
	testWs, testDemos = testData['ws'], testData['demos']
	rho = testData['demoBeta'] / POLICY_BETA
	print(f"Loaded {len(testWs)} test demonstrators from disk (demoBeta = {testData['demoBeta']})")

	results = []
	for user in range(len(testWs)):
		result = _evaluateUser(env, regions, models, stateSize, rho, user, testWs[user], testDemos[user])
		results.append(result)

	printCompareResults(results)
