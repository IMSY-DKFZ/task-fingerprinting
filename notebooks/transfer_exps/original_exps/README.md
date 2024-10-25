# Original Transfer Learning Experiments

> Note: To follow this reproduction you need to have access to a non-public repository inside DKFZ. Follow
> "recent_exps" for a public compatible reproduction.

All downstream experiments on top of neural network training can be reproduced via the instructions
given inside the "notebooks" folder.

## Pre-requisites

* requires conda, may be set up through [miniconda](https://docs.anaconda.com/free/miniconda/index.html)
* running experiments requires cuda capable GPU, evaluation does not
* running ALL experiments (> 30_000 trained models) likely requires a GPU cluster
* clone this repository

```commandline
cd your/path/to/reproduce/
git clone git@git.dkfz.de:imsy/ise/task-fingerprinting.git
```

## Experiments

The experiments were based on multiple version of the `mml` framework with backward compatibility breaking changes
inbetween as a lot of internal structure has been refactored. To make sure
that the Auto-Augment mode is still available (removed in b5a3d394a5fc2175215454727a25edbcae9b555f) but the multi-task
sampler balancing fix is available (fixed in a3bacbca763d02818ebff3591fc968960ed85424)
there is only a narrow range of versions. We recommend to check out: a3bacbca763d02818ebff3591fc968960ed85424

* Clone the mml repository and checkout

```commandline
cd your/path/to/reproduce/
git clone git@git.dkfz.de:imsy/ise/mml.git
cd mml
git checkout b5a3d394a5fc2175215454727a25edbcae9b555f
```

* create a conda environment using `conda.yaml` (next to this README) and install "old" `mml`

```commandline
cd ../task-fingerprinting/notebooks/transfer_exps/original_exps
conda update -n base -c defaults conda  # OPTIONAL: update conda
conda create -f conda.yaml  --name tf_original  # create environment
conda activate tf_original  # activate environment
cd your/path/to/reproduce/mml
pip install .  # install locally from cloned mml repo
```

* set env variables

```commandline
cd src/mml
cp example.env mml.env
nano mml.env
```

* adapt the following variables
    * MML_DATA_PATH # here mml will store the images, etc. of the tasks
    * MML_RESULTS_PATH # here mml will store results e.g. extracted features and models
    * MML_LOCAL_WORKERS # number of CPU cores you want to dedicate
* if running on the cluster is intended, also specify
    * MML_CLUSTER_DATA_PATH
    * MML_CLUSTER_RESULTS_PATH
    * MML_CLUSTER_WORKERS
* create another environment for the auto augment capability

```commandline
conda deactivate  # deactivate the other env
cd your/path/to/reproduce/
cd task-fingerprinting/notebooks/transfer_exps/original_exps
conda create -f conda.yaml --name tf_aa  # create environment
conda activate tf_aa  # activate environment
cd ../mml
pip install .[aa]
```

* no need to set environment variables again
* NOTE: use the `tf_original` environment as default for all experiments, only exception are the AutoAugment runs
* if running on the cluster you need to create those environments there as well, in addition set up two runner scripts
    * `mml.sh` that loads cuda, activates the `tf_original` conda env and forwards the args to `mml` adding
      `sys=cluster`
    * `aa.sh` that loads cuda, activates the `tf_aa` conda env and forwards the args to `mml` adding `sys=cluster`
* commands that need to be submitted (or run locally) are found in `notebooks/transfer_exps/original_exps`
    * the provided `output_XXX.txt` files are the ones originally used to run experiments
    * they need adaptation to the respective AD user or to run them locally, see `original.ipynb` for details
    * you may submit job batches as e.g.
      `ssh user@host 'bash -s' < notebooks/transfer_exps/original_exps/output_base.txt`
