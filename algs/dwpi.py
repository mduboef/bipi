import os
import pickle
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import (
	DEMO_BETA,
	DWPI_GRANULARITY, DWPI_N_EPISODES, DWPI_NDEMOS_TRAIN,
	DWPI_AUGMENT, DWPI_SF_GAMMA, DWPI_HIDDEN_DIM,
	DWPI_EPOCHS, DWPI_LR, DWPI_NDEMOS_INFER,
)
from helpers import runEpisode

ENCODINGS = ('return', 'stateFreq', 'sf')


# soft state value: (1/beta) * log sum_a exp(beta * Q(s,a))
def _softV(qVals, beta):
	x = beta * qVals
	m = x.max()
	return (1.0 / beta) * (m + np.log(np.sum(np.exp(x - m))))


# ─────────────────────────────────────────────────────────────
# weight space helpers
# ─────────────────────────────────────────────────────────────

# returns list of 2D weight vectors at given granularity: [w0, 1-w0] for w0 in {0, g, 2g, ..., 1}
def getWeightVecs(granularity=DWPI_GRANULARITY):
	steps = round(1.0 / granularity)
	return [np.array([round(i / steps, 10), round(1.0 - i / steps, 10)]) for i in range(steps + 1)]


# returns the weight vector in weightVecs closest (L2) to w
def findNearestWeight(w, weightVecs):
	dists = [float(np.linalg.norm(np.asarray(wv) - np.asarray(w))) for wv in weightVecs]
	return weightVecs[int(np.argmin(dists))]


# returns (nRows, nCols) from observation space upper bounds
def getGridDims(env):
	hi = env.observation_space.high
	return int(hi[0]) + 1, int(hi[1]) + 1


def _wKey(w):
	return (round(float(w[0]), 10), round(float(w[1]), 10))


# ─────────────────────────────────────────────────────────────
# DWMOTQ: one scalar Q-table per discretized weight
# ─────────────────────────────────────────────────────────────

# trains a separate epsilon-greedy tabular Q-agent for each weight in the discretized simplex
# Q-values are scalar (scalarized by w), not vector-valued
# returns dict: wKey -> {state_tuple -> np.array(nActions,)}
def trainDWMOTQ(env, granularity=DWPI_GRANULARITY, nEpisodes=DWPI_N_EPISODES, alpha=0.2, gamma=1.0):
	weightVecs = getWeightVecs(granularity)
	nActions = env.action_space.n
	qTables = {}

	for i, w in enumerate(weightVecs):
		key = _wKey(w)
		print(f"  DWMOTQ [{i+1:>3}/{len(weightVecs)}]  w = {np.round(w, 2)}", flush=True)
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

		qTables[key] = dict(qTable)

	return qTables


def saveDWMOTQ(qTables, saveDir):
	os.makedirs(saveDir, exist_ok=True)
	path = os.path.join(saveDir, 'dwmotq.pkl')
	with open(path, 'wb') as f:
		pickle.dump(qTables, f)
	print(f"  DWMOTQ saved → {path}")


def loadDWMOTQ(saveDir):
	with open(os.path.join(saveDir, 'dwmotq.pkl'), 'rb') as f:
		return pickle.load(f)


# returns the Q-table for the nearest discretized weight to w
def lookupQTable(qTables, w, weightVecs):
	nearest = findNearestWeight(w, weightVecs)
	return qTables[_wKey(nearest)]


# ─────────────────────────────────────────────────────────────
# demo generation with pure Boltzmann rationality (DEMO_BETA)
# ─────────────────────────────────────────────────────────────

# generates nDemos Boltzmann-rational episodes and returns their averaged encodings
def _encodeAvgBatch(env, qTable, nDemos, sfGamma, nRows, nCols):
	demoPolicy = makeBoltzmannDemoPolicy(qTable, DEMO_BETA)
	trajs = [runEpisode(env, demoPolicy)[0] for _ in range(nDemos)]
	retVec  = encodeDemos(trajs, 'return',    nRows, nCols, sfGamma)
	freqVec = encodeDemos(trajs, 'stateFreq', nRows, nCols, sfGamma)
	sfVec   = encodeDemos(trajs, 'sf',        nRows, nCols, sfGamma)
	return retVec, freqVec, sfVec


# returns a policy callable (obs, env) -> action that samples Boltzmann-rationally from a Q-table
def makeBoltzmannDemoPolicy(qTable, beta):
	def policy(obs, env):
		s = tuple(obs)
		if s not in qTable:
			return env.action_space.sample()
		logits = beta * qTable[s]
		logits = logits - logits.max()
		probs = np.exp(logits)
		probs /= probs.sum()
		return int(np.random.choice(len(probs), p=probs))
	return policy


# returns a policy callable (obs, env) -> action that always takes the greedy argmax action
def makeGreedyPolicy(qTable):
	def policy(obs, env):
		s = tuple(obs)
		if s not in qTable:
			return env.action_space.sample()
		return int(np.argmax(qTable[s]))
	return policy


# encodes pre-run trajectories (list of (obs, action, reward) tuples each) into a single feature vector
# used at DWPI inference time with trajectories collected from the environment
def encodeDemos(demos, encoding, nRows, nCols, sfGamma=DWPI_SF_GAMMA):
	nStates = nRows * nCols
	feats = []
	for traj in demos:
		if encoding == 'return':
			feat = np.zeros(2)
			for _, _, r in traj:
				feat += r
		elif encoding == 'stateFreq':
			feat = np.zeros(nStates)
			for obs, _, _ in traj:
				feat[int(obs[0]) * nCols + int(obs[1])] += 1.0
		elif encoding == 'sf':
			feat = np.zeros(nStates)
			for t, (obs, _, _) in enumerate(traj):
				feat[int(obs[0]) * nCols + int(obs[1])] += sfGamma ** t
		feats.append(feat)
	return np.mean(feats, axis=0).astype(np.float32)


# generates and returns the averaged test encodings for one user using the DWMOTQ agent
def generateTestEncodings(env, qTables, trueW, nDemos, sfGamma, weightVecs):
	nRows, nCols = getGridDims(env)
	qTable = lookupQTable(qTables, trueW, weightVecs)
	retVec, freqVec, sfVec = _encodeAvgBatch(env, qTable, nDemos, sfGamma, nRows, nCols)
	return {'return': retVec, 'stateFreq': freqVec, 'sf': sfVec}


# ─────────────────────────────────────────────────────────────
# training dataset construction
# ─────────────────────────────────────────────────────────────

# iterates over all discretized weights, generates augmentFactor averaged batches per weight
# returns (Xret, Xfreq, Xsf, Y) as float32 arrays
def buildDWPIDataset(env, qTables, granularity=DWPI_GRANULARITY, nDemos=DWPI_NDEMOS_TRAIN,
                     augmentFactor=DWPI_AUGMENT, sfGamma=DWPI_SF_GAMMA):
	weightVecs = getWeightVecs(granularity)
	nRows, nCols = getGridDims(env)
	total = len(weightVecs) * augmentFactor
	Xret, Xfreq, Xsf, Y = [], [], [], []
	count = 0

	for w in weightVecs:
		qTable = qTables[_wKey(w)]
		for _ in range(augmentFactor):
			retVec, freqVec, sfVec = _encodeAvgBatch(env, qTable, nDemos, sfGamma, nRows, nCols)
			Xret.append(retVec)
			Xfreq.append(freqVec)
			Xsf.append(sfVec)
			Y.append(w.copy())
			count += 1
			if count % 500 == 0:
				print(f"  dataset: {count}/{total}", flush=True)

	return (np.array(Xret,  dtype=np.float32),
	        np.array(Xfreq, dtype=np.float32),
	        np.array(Xsf,   dtype=np.float32),
	        np.array(Y,     dtype=np.float32))


def saveDataset(dataset, saveDir):
	os.makedirs(saveDir, exist_ok=True)
	path = os.path.join(saveDir, 'dataset.pkl')
	with open(path, 'wb') as f:
		pickle.dump(dataset, f)
	print(f"  dataset saved → {path}")


def loadDataset(saveDir):
	with open(os.path.join(saveDir, 'dataset.pkl'), 'rb') as f:
		return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# feedforward inference network (FNN)
# ─────────────────────────────────────────────────────────────

class _DWPIFNN(nn.Module):
	def __init__(self, inputDim, nObj, hiddenDim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(inputDim, hiddenDim),
			nn.ReLU(),
			nn.Linear(hiddenDim, hiddenDim),
			nn.ReLU(),
			nn.Linear(hiddenDim, nObj),
			nn.Softmax(dim=-1),
		)

	def forward(self, x):
		return self.net(x)


# trains a single FNN via MSE regression; X shape (N, inputDim), Y shape (N, nObj)
def trainDWPIModel(X, Y, nObj, hiddenDim=DWPI_HIDDEN_DIM, nEpochs=DWPI_EPOCHS, lr=DWPI_LR, batchSize=64):
	model = _DWPIFNN(X.shape[1], nObj, hiddenDim)
	opt = torch.optim.Adam(model.parameters(), lr=lr)
	lossFn = nn.MSELoss()
	loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
	                    batch_size=batchSize, shuffle=True)

	model.train()
	for epoch in range(nEpochs):
		epochLoss = 0.0
		for xb, yb in loader:
			pred = model(xb)
			loss = lossFn(pred, yb)
			opt.zero_grad()
			loss.backward()
			opt.step()
			epochLoss += loss.item() * len(xb)
		if (epoch + 1) % max(1, nEpochs // 4) == 0:
			print(f"  epoch {epoch+1}/{nEpochs}  mse={epochLoss / len(X):.6f}")

	model.eval()
	return model


def saveModels(models, saveDir):
	os.makedirs(saveDir, exist_ok=True)
	for name, model in models.items():
		path = os.path.join(saveDir, f'model_{name}.pt')
		torch.save(model.state_dict(), path)
		print(f"  model '{name}' saved → {path}")


# inputDims: dict of encodingName -> input feature dimension
def loadModels(inputDims, nObj, hiddenDim, saveDir):
	models = {}
	for name, inDim in inputDims.items():
		path = os.path.join(saveDir, f'model_{name}.pt')
		model = _DWPIFNN(inDim, nObj, hiddenDim)
		model.load_state_dict(torch.load(path, weights_only=True))
		model.eval()
		models[name] = model
	return models


# ─────────────────────────────────────────────────────────────
# inference
# ─────────────────────────────────────────────────────────────

# returns predicted weight as a numpy array given a pre-encoded feature vector
def inferWeight(model, featVec):
	with torch.no_grad():
		x = torch.from_numpy(featVec.reshape(1, -1))
		return model(x).numpy().squeeze()


# ─────────────────────────────────────────────────────────────
# results
# ─────────────────────────────────────────────────────────────

def printDWPIResults(userResults, regions):
	nUsers = len(userResults)
	print(f"\n  DWPI results ({nUsers} users):\n")

	correctCounts = {enc: 0 for enc in ENCODINGS}

	for res in userResults:
		trueW   = res['trueWeight']
		trueIdx = res['trueRegionIdx']
		r       = regions[trueIdx]
		print(f"  user {res['user']}:  w = {np.round(trueW, 4)}  (region {trueIdx}: "
		      f"wLeft={r['wLeft']:.4f} wRight={r['wRight']:.4f})")
		for enc in ENCODINGS:
			infW   = res['inferred'][enc]
			infIdx = res['inferredRegionIdx'][enc]
			ok     = 'correct' if infIdx == trueIdx else 'wrong'
			if infIdx == trueIdx:
				correctCounts[enc] += 1
			print(f"    {enc:<10}: ŵ = {np.round(infW, 4)}  → region {infIdx}  ({ok})")

	print(f"\n  accuracy summary:")
	for enc in ENCODINGS:
		acc = correctCounts[enc] / nUsers * 100
		print(f"    {enc:<10}: {correctCounts[enc]}/{nUsers}  ({acc:.1f}%)")
