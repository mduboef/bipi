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
from config import POLICY_BETA, DEMO_BETA, TRAJ_PER_USER, NUM_USERS
from helpers import makeEnv, printEnvInfo, runEpisode, findRegion, makeDemoPolicy, renderTrajectory
from algs.ccs import buildCCS, printCCS, saveCCS, loadCCS




def main():

	# stores names of environments to run experiments on
	envNames = ["deep-sea-treasure-v0"]
	methodNames = ["ccs", "bipi", "dwpi"]


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
		ccsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ccsResults', envName)
		ccs = loadCCS(ccsDir)
		if ccs['policyBeta'] != POLICY_BETA:
			print(f"Warning: CCS was computed with policyBeta={ccs['policyBeta']} but config has POLICY_BETA={POLICY_BETA}")
		regions = ccs['regions']
		
		nObj = len(regions[0]['returnVec'])

		for user in range(NUM_USERS):
			# randomly/unifromly sample latent prefence for user/sdemonstrator
			prefWeight = np.random.dirichlet(np.ones(nObj))
			print(f"\nUser {user}: true preference weight = {np.round(prefWeight, 4)}")

			# generate demonstrations
			# we cant just roll out the CCS policy that corresponds to prefWeight
			# that CCS policy (π*(a|s)) was trained with POLICY_BETA
			# By definition the demonstration should be a rationality of DEMO_BETA
			# substituting log π*(a|s) = POLICY_BETA · w·Q(s,a) + c(s) into P_demo gives:
			#   P_demo(a|s) ∝ π*(a|s)^ρ,  where ρ = DEMO_BETA / POLICY_BETA
			# so raising CCS policy probabilities to the power ρ and renormalizing gives the exact Boltzmann-rational demonstrator policy for DEMO_BETA
			rho = DEMO_BETA / ccs['policyBeta']								# rationality ratio (ρ)
			trueRegion = findRegion(prefWeight, regions)
			demoPolicy = makeDemoPolicy(trueRegion['policy'], rho)
			demos = [runEpisode(env, demoPolicy)[0] for _ in range(TRAJ_PER_USER)]
			print(f"  generated {TRAJ_PER_USER} demo(s), {sum(len(d) for d in demos)} total steps")

			# graphical rendering of the demos
			for i, demo in enumerate(demos):
				renderTrajectory(demo, f"demo {i}  (DEMO_BETA={DEMO_BETA})", envName)
			
			# rollout and render the optimal policy in the CCS for prefWeight
			optimalTraj, _ = runEpisode(env, makeDemoPolicy(trueRegion['policy'], rho=1.0))
			renderTrajectory(optimalTraj, f"optimal CCS policy rollout (POLICY_BETA={ccs['policyBeta']})", envName)

			# perform bipi on the demo data using the precomuted ccs with (volumes, optimalpolicies and expected returns for each region)
				# return a probability distribution over centroids representing their regions

			# get 4 different policies base on the distribution bipi spits out
				# 1. the pareto optimal policy associated with the region with the highest likelihood/probability
				# 2. the pareto optimal policy associated with the mean preference weight base on the ditribution
					# mean = \sum_{region} centroid_{region} * prob of region
				# 3. the pareto optimal policy that results in the highest expected utility considering the likelihood of every region
				# 4. the pareto optimal policy with the best 5% CVaR considering the distribution over possible preference weights

			# evaluate bipi's performance
				# was the real preference actually in the predicted region?
				# what is the expected utility of the user if they follow the policy we assigned to them?



	# else:	# dwpi
	# 	# try to load ccs from saved files


	env.close()

main()