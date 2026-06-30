# episodes used for CCS Q-learning (buildCCS + trainSoftmaxPolicies) per environment
# add an entry here before running any mode on a new environment
CCS_EPISODES = {
	"deep-sea-treasure-v0": 70000,
	"fishwood-v0":          40000,
}

# discount factor for CCS Q-learning per environment
# use gamma < 1 for environments that only truncate (never terminate naturally) to avoid
# the gamma=1 convergence trap where self-referential Bellman updates give wrong action ordering
CCS_GAMMA = {
	"deep-sea-treasure-v0": 1.0,   # terminates naturally at treasure; gamma=1 is correct
	"fishwood-v0":          0.99,  # only truncates at 200 steps; cycling states need gamma < 1
}

POLICY_BETA = 20.0
DEMO_BETA = 20.0

TRAJ_PER_USER = 100			# how many trajectories worth of demo data should each demonstrator generate
NUM_USERS = 1000			# how many unique demonstrators/users should we simulate

# DWPI hyperparameters (Lu et al. 2024)
DWPI_GRANULARITY  = 0.01	# simplex discretization granularity (g in paper; 101 weights for 2D)
DWPI_N_EPISODES   = 10000	# Q-learning episodes per weight during DWMOTQ training
DWPI_NDEMOS_TRAIN = 100		# episodes averaged per training example
DWPI_AUGMENT      = 50		# training examples generated per weight (augmentation factor)
DWPI_SF_GAMMA     = 0.99	# discount factor used when computing successor feature encoding
DWPI_HIDDEN_DIM   = 64		# hidden layer width for FNN inference model
DWPI_EPOCHS       = 200		# FNN training epochs
DWPI_LR           = 1e-3	# Adam learning rate
DWPI_NDEMOS_INFER = 100		# episodes averaged per user at inference time


