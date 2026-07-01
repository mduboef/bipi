import os
import pickle
import numpy as np
from colorama import Fore
from collections import defaultdict
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport


# tabular MO Q-learning at a fixed scalarization weight
# uses hard (greedy) Bellman backup so Q-values converge to Q* for the deterministic optimal policy
# returns qTable: {state_tuple -> np.array shape (nActions, nObj)}
def moQLearning(env, weight, nObj, nEpisodes=10000, alpha=0.1, gamma=1.0, epsilonStart=1.0, epsilonEnd=0.01):
	nActions = env.action_space.n
	qTable = defaultdict(lambda: np.zeros((nActions, nObj)))

	for ep in range(nEpisodes):
		eps = epsilonStart + (epsilonEnd - epsilonStart) * (ep / nEpisodes)
		obs, _ = env.reset()
		s = tuple(obs)
		done = False

		while not done:
			if np.random.random() < eps:
				a = env.action_space.sample()
			else:
				a = int(np.argmax(qTable[s] @ weight))

			nextObs, r, terminated, truncated, _ = env.step(a)
			ns = tuple(nextObs)
			done = terminated or truncated

			if done:
				target = r
			else:
				aStar = int(np.argmax(qTable[ns] @ weight))
				target = r + gamma * qTable[ns][aStar]

			qTable[s][a] += alpha * (target - qTable[s][a])
			s = ns

	return dict(qTable)


# converts Q-table to deterministic policy: {state -> one-hot action probability array}
# optimal action at each state is argmax over scalarized Q-values
def qTableToPolicy(qTable, weight):
	nActions = next(iter(qTable.values())).shape[0]
	policy = {}
	for s, qVecs in qTable.items():
		probs = np.zeros(nActions)
		probs[int(np.argmax(qVecs @ weight))] = 1.0
		policy[s] = probs
	return policy


# estimates J^pi at the initial state using the greedy action's Q-vector at s0
def computeReturnVec(qTable, s0, weight, nObj):
	if s0 not in qTable:
		return np.zeros(nObj)
	aStar = int(np.argmax(qTable[s0] @ weight))
	return qTable[s0][aStar]


# attaches wLeft, wRight, volume, centroid to each entry in-place (2D only)
# entries must be sorted by returnVec[0] ascending before calling
def _attachRegions2D(entries):
	bps = [0.0]
	for i in range(len(entries) - 1):
		rv_i = entries[i]['returnVec']
		rv_j = entries[i + 1]['returnVec']
		denom = (rv_i[0] - rv_j[0]) + (rv_j[1] - rv_i[1])
		wStar = (rv_j[1] - rv_i[1]) / denom if abs(denom) > 1e-12 else 0.5
		bps.append(float(np.clip(wStar, 0.0, 1.0)))
	bps.append(1.0)

	for i, e in enumerate(entries):
		wL, wR = bps[i], bps[i + 1]
		mid = (wL + wR) / 2.0
		e['wLeft'] = wL
		e['wRight'] = wR
		e['volume'] = wR - wL
		e['centroid'] = np.array([mid, 1.0 - mid])


# outer loop: LinearSupport + inner tabular MO Q-learning, followed by centroid retraining
# returns CCS: list of dicts with keys policy, qTable, returnVec, trainingWeight, wLeft, wRight, volume, centroid
def buildCCS(env, epsilon=1e-4, nEpisodes=10000, alpha=0.1, gamma=1.0):
	nObj = env.unwrapped.reward_space.shape[0]
	if nObj != 2:
		raise NotImplementedError("region computation only supported for 2 objectives")

	obs, _ = env.reset()
	s0 = tuple(obs)

	ls = LinearSupport(num_objectives=nObj, epsilon=epsilon, verbose=False)
	ccsEntries = []

	w = ls.next_weight()
	while w is not None:
		print(f"  training w = {np.round(w, 4)} ...")
		qTable = moQLearning(env, w, nObj, nEpisodes=nEpisodes, alpha=alpha, gamma=gamma)
		policy = qTableToPolicy(qTable, w)
		rv = computeReturnVec(qTable, s0, w, nObj)

		nBefore = len(ls.ccs)
		removedIdxs = ls.add_solution(rv, w)

		validRemoved = [i for i in removedIdxs if i < nBefore]
		for idx in sorted(validRemoved, reverse=True):
			ccsEntries.pop(idx)

		if len(ls.ccs) == nBefore - len(validRemoved) + 1:
			ccsEntries.append({'policy': policy, 'qTable': qTable, 'returnVec': rv, 'trainingWeight': w})

		w = ls.next_weight()

	# deduplicate: remove entries with nearly identical return vectors
	seen = []
	dedupEntries = []
	for e in ccsEntries:
		if not any(np.allclose(e['returnVec'], rv, atol=0.05) for rv in seen):
			seen.append(e['returnVec'])
			dedupEntries.append(e)

	# warn if any known Pareto-optimal policy is absent from the discovered CCS
	try:
		paretoFront = env.unwrapped.pareto_front(gamma=gamma)
		for trueRv in paretoFront:
			if not any(np.allclose(trueRv, e['returnVec'], atol=0.5) for e in dedupEntries):
				print(f"Warning: Pareto-optimal policy {np.round(trueRv, 4)} not found in CCS")
	except AttributeError:
		pass

	dedupEntries.sort(key=lambda e: e['returnVec'][0])
	_attachRegions2D(dedupEntries)
	return dedupEntries


# prints a formatted summary of the CCS
def printCCS(ccs):
	print(f"  CCS ({len(ccs)} policies):")
	print(f"  {'rv[0]':>8}  {'rv[1]':>6}  {'w_left':>8}  {'w_right':>8}  {'volume':>8}  centroid")
	for e in ccs:
		rv, c = e['returnVec'], e['centroid']
		print(f"  {rv[0]:>8.4f}  {rv[1]:>6.4f}  {e['wLeft']:>8.4f}  {e['wRight']:>8.4f}  {e['volume']:>8.4f}  [{c[0]:.4f}, {c[1]:.4f}]")
	print()


# saves CCS results to disk under saveDir/ccs.pkl
# payload includes one dict per region with: policy, qTable, returnVec, wLeft, wRight, volume, centroid
def saveCCS(ccs, saveDir):
	os.makedirs(saveDir, exist_ok=True)
	payload = {
		'regions': [
			{
				'policy':    e['policy'],
				'returnVec': e['returnVec'],
				'wLeft':     e['wLeft'],
				'wRight':    e['wRight'],
				'volume':    e['volume'],
				'centroid':  e['centroid'],
			}
			for e in ccs
		]
	}
	savePath = os.path.join(saveDir, 'ccs.pkl')
	with open(savePath, 'wb') as f:
		pickle.dump(payload, f)
	print(f"  CCS saved → {savePath}")


# loads a previously saved CCS from saveDir/ccs.pkl
# returns {'regions': list of dicts}
def loadCCS(saveDir):
	loadPath = os.path.join(saveDir, 'ccs.pkl')
	with open(loadPath, 'rb') as f:
		return pickle.load(f)


# converts Q-table to softmax policy: {state -> prob array}
# action probs are proportional to exp(beta * Q(s,a) @ weight)
def qTableToSoftmaxPolicy(qTable, weight, beta):
	policy = {}
	for s, qVecs in qTable.items():
		logits = beta * (qVecs @ weight)
		logits -= logits.max()
		probs = np.exp(logits)
		probs /= probs.sum()
		policy[s] = probs
	return policy


# trains one softmax policy per CCS region at that region's centroid weight
# returns list of dicts: {regionIdx, centroid, policy}
def trainSoftmaxPolicies(env, ccs, beta, nEpisodes=10000, alpha=0.1, gamma=1.0):
	nObj = env.unwrapped.reward_space.shape[0]
	softmaxPolicies = []
	for k, region in enumerate(ccs):
		centroid = region['centroid']
		print(f"  region {k}: training at centroid {np.round(centroid, 4)} ...")
		qTable = moQLearning(env, centroid, nObj, nEpisodes=nEpisodes, alpha=alpha, gamma=gamma)
		policy = qTableToSoftmaxPolicy(qTable, centroid, beta)
		softmaxPolicies.append({'regionIdx': k, 'centroid': centroid, 'policy': policy})
	return softmaxPolicies


# rolls out each region's greedy softmax policy and warns if it fails to reach its own returnVec
# BIPI's likelihood assumes each region policy reaches that region's treasure; if a deep-treasure
# policy is under-converged and stops short, BIPI cannot recognize demonstrators for that region
def verifyRegionPolicies(env, ccs, softmaxPolicies, maxSteps=200, atol=0.05):
	policies = {sp['regionIdx']: sp['policy'] for sp in softmaxPolicies}
	print("  verifying region policies reach their own returnVec ...")
	for k, region in enumerate(ccs):
		policy = policies.get(k, {})
		obs, _ = env.reset()
		total = np.zeros(len(region['returnVec']))
		done = False
		steps = 0
		while not done and steps < maxSteps:
			s = tuple(obs)
			a = int(np.argmax(policy[s])) if s in policy else env.action_space.sample()
			obs, r, terminated, truncated, _ = env.step(a)
			total += r
			done = terminated or truncated
			steps += 1
		reached = abs(total[0] - region['returnVec'][0]) <= atol
		flag = "" if reached else "  <-- WARNING: does not reach its own treasure"
		print(f"    region {k}: expected {np.round(region['returnVec'], 2)}  reached {np.round(total, 2)}{flag}")


# saves softmax policies to saveDir/softmaxPolicies.pkl
def saveSoftmaxPolicies(softmaxPolicies, saveDir):
	os.makedirs(saveDir, exist_ok=True)
	savePath = os.path.join(saveDir, 'softmaxPolicies.pkl')
	with open(savePath, 'wb') as f:
		pickle.dump(softmaxPolicies, f)
	print(f"  softmax policies saved → {savePath}")


# loads softmax policies from saveDir/softmaxPolicies.pkl
# returns list of dicts: {regionIdx, centroid, policy}
def loadSoftmaxPolicies(saveDir):
	loadPath = os.path.join(saveDir, 'softmaxPolicies.pkl')
	with open(loadPath, 'rb') as f:
		return pickle.load(f)
