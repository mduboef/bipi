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


## Issues

The results are really bad for bipi. My main instinct is the even though DWPI's inference networks relies of those "low-fidelity" representations of trajectories (ie. endoing a trajectory as a vector of returns) it is being helped by the fact that it has SO MANY "datapoints" throughout the policy space compared to BIPI which only has 1 soft-max policy per region, with that 1 softmax policy. We are looking the liklihood that that one particalr centroid softmax policy would have produced the demonstrators observed action action and trying to claim that that says something about the likelihood the demonstrator's latent preference falls into that region of preference space. My instinct is that looking at a trajectory action by action gives a more "high fidelity" imagine of the underlying demonstrator's policy, but BIPI is being limited by the fact that it has so few reference points (softmax policies) to compare the trajectory against. DWPI may have a "lower fidelity" image of the trajectory but it has the advantage of comparing that image against thousands of other "lower fidelity" images spread across the preference spaces (the training data used to train the preference inference network). I also think a stochastic environment would bring out the limitations of DWPI's low fidelity approach.