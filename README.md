# Bayesian Inverse Preference Inference (BIPI)

Infers a demonstrator's latent preferences over multiple objectives from demonstration data and assigns them the Pareto-optimal policy best suited to those preferences.

Final project for CS 690S AI Alignment.

## Overview

Given a precomputed **Convex Coverage Set (CCS)** of Pareto-optimal softmax policies and a short demonstration, BIPI looks through the demo action-by-action updating a posterior over various regions of the preference space, where all weights in a region correspond to the same Pareto-optimal policy. Likelihood updates use only stored policy probabilities, no Q-values, no RL at inference time.

The approach supports four policy selection strategies (MAP, mean weight, max expected utility, CVaR) and is compared head-to-head against **DWPI** (Lu et al. 2024) on `DeepSeaTreasure-v0` (convex map, Yang et al. 2019).

## Demo Generation

Demonstrations are generated synthetically using **DWMOTQ**, a collection of separate tabular Q-tables, one trained per discretized weight vector on the simplex (101 tables at granularity g=0.01 for 2 objectives). Each Q-table is trained independently via epsilon-greedy Q-learning with the reward scalarized by its weight, so its Q-values approximate the optimal returns for every (state, action) pair under that specific preference.

To generate a demo for a given preference weight w:
1. Snap w to the nearest discretized weight and retrieve that Q-table.
2. Apply the Boltzmann policy with `DEMO_BETA` over the Q-values — this gives a stochastic policy with non-zero probability for all actions.
3. Roll out the policy to collect (state, action) trajectories labeled with w.

DWPI uses a large dataset of such demos (across all discretized weights) to train its FNN inference models. BIPI uses the same demo generation at evaluation time to simulate test users.

## Setup

```bash
pip install mo-gymnasium morl-baselines
```

## Usage

Run modes in order — each depends on artifacts from the previous step.

```bash
python3 main.py deep-sea-treasure-v0 ccs      # compute and save the CCS
python3 main.py deep-sea-treasure-v0 dwpi     # train DWMOTQ, build dataset, train inference models
python3 main.py deep-sea-treasure-v0 bipi     # run BIPI on simulated users
python3 main.py deep-sea-treasure-v0 compare  # head-to-head BIPI vs DWPI evaluation
```

Append `-render` to the `bipi` command to visualize demos and policy rollouts.

## Project Structure

```
algs/
  ccs.py       — CCS computation (soft MO Q-learning + LinearSupport outer loop)
  bipi.py      — BIPI inference algorithm and policy selection strategies
  dwpi.py      — DWPI baseline (DWMOTQ training, dataset, FNN inference)
compare.py     — head-to-head comparison runner
main.py        — entry point and mode dispatch
config.py      — all hyperparameters
helpers.py     — environment utilities
bipiFormalization.tex  — algorithm formalization and proofs
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `POLICY_BETA` | 20.0 | Boltzmann rationality of CCS policies |
| `DEMO_BETA` | 20.0 | Boltzmann rationality of the demonstrator |
| `TRAJ_PER_USER` | 1 | Demonstration trajectories per simulated user |
| `NUM_USERS` | 100 | Number of simulated users evaluated per run |

## References

- Lu et al. (2024). *DWPI: Inferring Reward Functions of Multi-Objective Markov Decision Processes.* Neural Computing and Applications.
- Alegre et al. (2023). *Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization.* AAMAS.


## The Big Fucking Unsolved Issue

Actual pareto optimal policies. For each policy there is a set of preference weights for which that policy is completely optimal.

Softmax policies. Very similar to pareto optimal policies but not 100% optimal they are boltzmann rational (POLICY_BETA is very high). This suboptimality is needed so that all actions have non-zero probability.
  Many questions about how we calculate these:
    These policies must have a single preference weight associated with them. The policy probabilites are determined by the untility of the actions which differs between pref weights in the same region (correspond the the same strickly optimal policy)
      What are we using? The centroid?
    When doing inference can we take into account the difference between the candiate weight and the centroid? This might allow us to account for the fact the softmax policy we have stored (one for the whole region, probably the centroid) might be very different from the softmax that WOULD be assciated with the actual candiate weight (if we had that softmax policy)
    
We are comparing every demo, which actually represents a single preference weight, to the policy for a centroids used to generate the softmax policy we have saved. We are assuming that the centroid policy that was most likely to generate/explain the demo is the same as the liklihood that the demonstrators single preference weight falls within the centroid's region. This is not true and fucking us up. Just because the centroid softmax policy is nearly identical to the pareto optimal policy doesn't mean that the centroid's small probability nuances are representative of the region as a whole.

Also I think I'm using the word CCS wrong. I've been calling the set of softmax nearly optimal policies as the CCS or pareto front. That is not strictly accurate and I think is leading to confusion.