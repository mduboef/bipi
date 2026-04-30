# action index -> human-readable label
ACTION_LABELS = {0: "up", 1: "down", 2: "left", 3: "right"}

# reward vector indices
TREASURE_IDX = 0
TIME_IDX = 1

POLICY_BETA = 20.0
DEMO_BETA = 5.0

TRAJ_PER_USER = 1		# how many trajectories worth of demo data should each demonstrator generate
NUM_USERS = 1			# how many unique demonstrators/users should we simulate


