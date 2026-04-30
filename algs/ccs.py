import os
import pickle
import numpy as np
from collections import defaultdict
from morl_baselines.multi_policy.linear_support.linear_support import LinearSupport
from config import POLICY_BETA


# soft Bellman backup: expected Q-vector under the softmax policy at next state
# this ensures Q-values are consistent with the Boltzmann rationality assumption
def _softTarget(qVecs, weight, betaTrain):
	u = qVecs @ weight
	u = u - u.max()
	probs = np.exp(betaTrain * u)
	probs /= probs.sum()
	return probs @ qVecs  # expected Q-vector under softmax policy


# tabular soft MO Q-learning at a fixed scalarization weight
# uses soft Bellman backup so Q-values are consistent with the softmax policy π_β
# returns qTable: {state_tuple -> np.array shape (nActions, nObj)}
def moQLearning(env, weight, nObj, betaTrain, nEpisodes=10000, alpha=0.1, gamma=1.0, epsilonStart=1.0, epsilonEnd=0.01):
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
				target = r + gamma * _softTarget(qTable[ns], weight, betaTrain)

			qTable[s][a] += alpha * (target - qTable[s][a])
			s = ns

	return dict(qTable)


# converts Q-table to softmax policy: {state -> action probability array}
def qTableToPolicy(qTable, weight, betaTrain):
	policy = {}
	for s, qVecs in qTable.items():
		u = qVecs @ weight
		u = u - u.max()  # shift for numerical stability before exp
		expU = np.exp(betaTrain * u)
		policy[s] = expU / expU.sum()
	return policy


# estimates J^pi at the initial state using Q-table values and softmax weights
def computeReturnVec(qTable, s0, weight, betaTrain, nObj):
	if s0 not in qTable:
		return np.zeros(nObj)
	u = qTable[s0] @ weight
	u = u - u.max()
	expU = np.exp(betaTrain * u)
	probs = expU / expU.sum()
	return probs @ qTable[s0]


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


# outer loop: LinearSupport + inner tabular MO Q-learning
# returns CCS: list of dicts with keys policy, returnVec, trainingWeight, wLeft, wRight, volume, centroid
def buildCCS(env, betaTrain=POLICY_BETA, epsilon=1e-4, nEpisodes=10000, alpha=0.1, gamma=1.0):
	nObj = env.unwrapped.reward_space.shape[0]
	if nObj != 2:
		raise NotImplementedError("region computation only supported for 2 objectives")

	obs, _ = env.reset()
	s0 = tuple(obs)

	ls = LinearSupport(num_objectives=nObj, epsilon=epsilon, verbose=False)
	ccsEntries = []  # mirrors ls.ccs exactly — only surviving non-dominated policies

	w = ls.next_weight()
	while w is not None:
		print(f"  training w = {np.round(w, 4)} ...")
		qTable = moQLearning(env, w, nObj, betaTrain, nEpisodes=nEpisodes, alpha=alpha, gamma=gamma)
		policy = qTableToPolicy(qTable, w, betaTrain)
		rv = computeReturnVec(qTable, s0, w, betaTrain, nObj)

		nBefore = len(ls.ccs)
		removedIdxs = ls.add_solution(rv, w)

		# remove dominated entries (indices < nBefore are real removals; == nBefore is the sentinel
		# returned when the new value itself is dominated and was not added)
		validRemoved = [i for i in removedIdxs if i < nBefore]
		for idx in sorted(validRemoved, reverse=True):
			ccsEntries.pop(idx)

		# append only if the new value was actually added to ls.ccs
		if len(ls.ccs) == nBefore - len(validRemoved) + 1:
			ccsEntries.append({'policy': policy, 'returnVec': rv, 'trainingWeight': w})

		w = ls.next_weight()

	# deduplicate: LinearSupport doesn't reject equal return vectors
	# so two training runs converging to the same policy can produce identical entries in ls.ccs
	seen = []
	dedupEntries = []
	# here we are removing any policies that have nearly identical reward vecs
	# ? Why not just check if they are exactly the same? Should the margin be tighter so we don't throw away a distinct region
	for e in ccsEntries:
		if not any(np.allclose(e['returnVec'], rv, atol=0.05) for rv in seen):
			seen.append(e['returnVec'])
			dedupEntries.append(e)

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
# payload includes policyBeta and one dict per region with: policy, returnVec, volume, centroid
def saveCCS(ccs, saveDir, beta):
	os.makedirs(saveDir, exist_ok=True)
	payload = {
		'policyBeta': beta,
		'regions': [
			{
				'policy': e['policy'],
				'returnVec': e['returnVec'],
				'wLeft': e['wLeft'],
				'wRight': e['wRight'],
				'volume': e['volume'],
				'centroid': e['centroid'],
			}
			for e in ccs
		]
	}
	savePath = os.path.join(saveDir, 'ccs.pkl')
	with open(savePath, 'wb') as f:
		pickle.dump(payload, f)
	print(f"  CCS saved → {savePath}  (policyBeta={beta})")


# loads a previously saved CCS from saveDir/ccs.pkl
# returns {'policyBeta': float, 'regions': list of dicts}
def loadCCS(saveDir):
	loadPath = os.path.join(saveDir, 'ccs.pkl')
	with open(loadPath, 'rb') as f:
		return pickle.load(f)
