# Experimental notebooks

> Note: Every notebook starts with an `mml.interactive.init()` that you might need to provide with the path to your 
> `mml.env` file.

## Overview

The `transfer_exps` directory contains the experimental descriptions and instructions to simulate knowledge transfer.
The other notebooks (numerated from `0` to `12`) investigate the knowledge transfer predictions (which we call
`TaskDistances`) and evaluate them. More precisely:

* `0_fill_cache.ipynb` - extracts transfer experiment outcomes and pre-computes task distances
* `1_develop_comparison.ipynb` - our full "tuning" procedure on the development tasks
* `2_internal_validation.ipynb` - evaluates the three bKLD variants compared to manual selection
* `3_external_validation.ipynb` - evaluates bKLD compared to previous task dsiatance measures
* `4_vary_bins.ipynb` - explores the impact of `b` (number of bins) on bKLD
* `5_vary_samples.ipynb` - explores the impact of `n` (number of samples) on bKLD
* `6_vary_hyperparameters.ipynb` - explores the impact of further modifications on task distances
* `7_full_sized_exps.ipynb` - additionally evaluates task distances on non-shrunk target tasks
* `8_tasks_infos.ipynb` - extracts task information
* `9_complexity.ipynb` - side note on feature strength correlation with task complexity
* `10_scenario_similarity.ipynb` - compares the optimal task distances across transfer scenarios
* `11_model_infos.ipynb` - extracts information on the architectures we used
* `12_fingerprint_visual.ipynb` - visualizations of fingerprinting space

## Usage

The `cache` (see `0_fill_cache.ipynb`) is provided within this repository and does not need to be re-computed. Hence,
all other notebooks can be run independently. The produced figures we used for the paper (see notebooks `2`, `3` & `5`)
are stored in `figures` - alongside the extracted `model_infos.csv` and `advanced_task_infos.csv`.

The intermediate computations in `6_vary_hyperparameters.ipynb` are optional to be skipped, as they are
stored in the `cache` as well.
