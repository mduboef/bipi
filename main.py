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

import sys
import numpy as np
from config import ACTION_LABELS, TREASURE_IDX, TIME_IDX, POLICY_BETA, DEMO_BETA
from helpers import makeEnv, printCCSInfo, runEpisode
from algs.ccs import buildCCS, printCCS




def main():

	# stores names of environments to run experiments on
	envNames = ["deep-sea-treasure-v0"]

	if sys.argv[1] not in envNames:
		print(f"Usage: python main.py <envName>")
		print(f"  where envName is one of: {envNames}")
		sys.exit(1)
	else: envName = sys.argv[1]

	env = makeEnv()

	printCCSInfo(env, envName)

	print(f"Calculating CCS (beta = {POLICY_BETA}) ...")
	# trajectory, totalReward = runEpisode(env, randomPolicy)
	# print(f"  steps            : {len(trajectory)}")
	# print(f"  total reward     : treasure={totalReward[TREASURE_IDX]:.1f}, time={totalReward[TIME_IDX]:.0f}")
	# print(f"  final state      : {trajectory[-1][0]}")
	
	ccs = buildCCS(env, nEpisodes=100000)
	printCCS(ccs)

	env.close()

main()