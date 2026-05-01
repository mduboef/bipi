import numpy as np


def _logsumexp(x):
	m = x.max()
	return m + np.log(np.sum(np.exp(x - m)))


# run BIPI algorithm and return the posterior over CCS regions
# regions : list of CCS region dicts (policy, returnVec, volume, centroid, wLeft, wRight)
# demos   : list of trajectories, each a list of (obs, action, reward) tuples
# rho     : rationality ratio = DEMO_BETA / POLICY_BETA
def runBIPI(regions, demos, rho):
	K = len(regions)
	volumes = np.array([r['volume'] for r in regions])

	# phase 1: log prior proportional to region volume — larger regions get more prior mass
	logScores = np.log(volumes)

	# phase 2: sequential Bayesian update
	# for each observed (s_t, a_t), accumulate the log-likelihood under each region's policy:
	#   λ_{k,t} = ρ · log π_k(a_t|s_t) − logsumexp(ρ · log π_k(·|s_t))
	# this is the log probability that a Boltzmann-rational demonstrator with rationality ρ
	# would choose action a_t in state s_t if their true preference is in region k
	for traj in demos:
		for obs, action, _ in traj:
			s = tuple(obs)
			for k, region in enumerate(regions):
				if s not in region['policy']:
					continue
				logProbs = np.log(np.clip(region['policy'][s], 1e-300, None))
				logScores[k] += rho * logProbs[action] - _logsumexp(rho * logProbs)

	# phase 3: normalize via log-sum-exp for numerical stability
	posterior = np.exp(logScores - _logsumexp(logScores))
	return posterior


# MAP: assign the policy of the highest-probability region
def selectMapPolicy(posterior):
	return int(np.argmax(posterior))


# Mean Weight: compute the posterior mean weight wMean = Σ_k P(R_k|τ)·c_k, then assign the policy whose region contains wMean
def selectMeanWeightPolicy(posterior, regions):
	centroids = np.array([r['centroid'] for r in regions])
	wMean = centroids.T @ posterior
	w = float(wMean[0])
	for i, r in enumerate(regions):
		if r['wLeft'] <= w <= r['wRight']:
			return i
	return len(regions) - 1


# Max Expected Utility: assign the policy that maximizes Σ_j P(R_j|τ) · c_j^T · J^{π_k}, equivalent to argmax_k wMean · J^{π_k}
def selectEUPolicy(posterior, regions):
	centroids  = np.array([r['centroid']  for r in regions])
	returnVecs = np.array([r['returnVec'] for r in regions])
	wMean = centroids.T @ posterior
	return int(np.argmax(returnVecs @ wMean))


# CVaR: assign the policy with the best worst-case expected utility
# for each candidate policy π_k, sort regions by ascending utility c_j^T · J^{π_k},
# accumulate posterior mass until it reaches alpha, take the probability-weighted average
# utility over that tail, this is CVaR_alpha(π_k)
def selectCVaRPolicy(posterior, regions, alpha=0.05):
	K = len(regions)
	centroids  = np.array([r['centroid']  for r in regions])
	returnVecs = np.array([r['returnVec'] for r in regions])

	cvarScores = np.zeros(K)
	for k in range(K):
		utilities  = centroids @ returnVecs[k]
		sortedIdxs = np.argsort(utilities)       # ascending: worst utility first
		cumMass  = 0.0
		cvarSum  = 0.0
		for j in sortedIdxs:
			if cumMass >= alpha:
				break
			weight   = min(posterior[j], alpha - cumMass)  # partial contribution at boundary
			cvarSum += weight * utilities[j]
			cumMass += weight
		cvarScores[k] = cvarSum / alpha
	return int(np.argmax(cvarScores))


# prints a formatted summary of the posterior and all four policy selections
def printBIPIResults(posterior, regions, trueRegionIdx, prefWeight, mapIdx, meanIdx, euIdx, cvarIdx, cvarAlpha=0.05):
	K = len(regions)
	centroids = np.array([r['centroid'] for r in regions])
	wMean = centroids.T @ posterior

	print(f"\n  posterior over {K} regions:")
	print(f"  {'k':>4}  {'wLeft':>7}  {'wRight':>7}  {'P(R_k|τ)':>10}  notes")
	for k in range(K):
		notes = []
		if k == trueRegionIdx: notes.append("true")
		if k == mapIdx:        notes.append("MAP")
		if k == meanIdx:       notes.append("mean-w")
		if k == euIdx:         notes.append("EU")
		if k == cvarIdx:       notes.append(f"CVaR({cvarAlpha})")
		print(f"  {k:>4}  {regions[k]['wLeft']:>7.4f}  {regions[k]['wRight']:>7.4f}  {posterior[k]:>10.4f}  {', '.join(notes)}")

	print(f"\n  true preference : w = {np.round(prefWeight, 4)}  (region {trueRegionIdx})")
	print(f"  posterior mean  : w = {np.round(wMean, 4)}")
	print(f"\n  policy selections:")
	print(f"    MAP           : region {mapIdx:>3}  ({'correct' if mapIdx  == trueRegionIdx else 'wrong'})")
	print(f"    mean weight   : region {meanIdx:>3}  ({'correct' if meanIdx == trueRegionIdx else 'wrong'})")
	print(f"    EU            : region {euIdx:>3}  ({'correct' if euIdx   == trueRegionIdx else 'wrong'})")
	print(f"    CVaR({cvarAlpha})     : region {cvarIdx:>3}  ({'correct' if cvarIdx == trueRegionIdx else 'wrong'})")


# prints an accuracy summary across all users for each policy selection method
def printBIPIAccuracy(userResults):
	nUsers   = len(userResults)
	methods  = ['map', 'mean', 'eu', 'cvar']
	labels   = {'map': 'MAP', 'mean': 'mean weight', 'eu': 'EU', 'cvar': 'CVaR(0.05)'}
	counts   = {m: sum(1 for r in userResults if r[m] == r['trueRegionIdx']) for m in methods}

	print(f"\n  accuracy summary:")
	for m in methods:
		acc = counts[m] / nUsers * 100
		print(f"    {labels[m]:<14}: {counts[m]}/{nUsers}  ({acc:.1f}%)")
