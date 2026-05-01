# Bayesian Inverse Preference Inference (BIPI)

Infers a demonstrator's latent preferences over multiple objectives from demonstration data and assigns them the Pareto-optimal policy based on the inferred probability distribution over preferences.

This is my final project for CS 690S AI Alignment.

## Overview

Given a precomputed **Convex Coverage Set (CCS)** of Pareto-optimal policies and a short demonstration, BIPI maintains a Bayesian posterior over preference weight regions on the simplex. Each region corresponds to one pareto optimal policy. Likelihood updates require only stored policy probabilities, no Q-values. There is no RL training at inference time.

The approach is compared against **DWPI** (Lu et al. 2024) on the `DeepSeaTreasure-v0` environment (convex map, Yang et al. 2019).

## Setup

```bash
pip install mo-gymnasium morl-baselines
```

## Usage

```bash
python3 main.py deep-sea-treasure-v0 ccs
```
This computes the CCS via tabular soft MO Q-learning + LinearSupport and saves the result to `ccsResults/`.

## Project Structure

```
algs/
  ccs.py       — CCS computation (MO Q-learning + LinearSupport outer loop)
  bipi.py      — BIPI inference algorithm
  dwpi.py      — DWPI baseline algorithm
main.py        — entry point
config.py      — hyperparameters (policy β, demo β)
helpers.py     — environment utilities
bipiFormalization.tex  — algorithm formalization and proofs
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `POLICY_BETA` | ~∞ | Boltzmann rationality of CCS policies |
| `DEMO_BETA` | 20.0 | Boltzmann rationality of the demonstrator |

## References

- Lu et al. (2024). *DWPI: Inferring Reward Functions of Multi-Objective Markov Decision Processes.* Neural Computing and Applications.
- Alegre et al. (2023). *Sample-Efficient Multi-Objective Learning via Generalized Policy Improvement Prioritization.* AAMAS.
