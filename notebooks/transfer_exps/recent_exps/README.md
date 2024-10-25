# Public Reproduction of Transfer Learning Experiments

> Note: Previous versions of the underlying software framework to run the training experiments "Medical Meta Learner"
> (MML), are not public. To reproduce the training experiments with a recent version of MML follow instructions inside
> "recent_exps". The original run commands are provided within "original_exps" for transparency. Though they are not
> fully compatible with recent MML.

All downstream experiments on top of neural network training can be reproduced via the instructions
given inside the "notebooks" folder.

## Pre-requisites

* requires conda, may be set up through [miniconda](https://docs.anaconda.com/free/miniconda/index.html)
* running experiments requires cuda capable GPU, evaluation does not
* running ALL experiments (> 30_000 trained models) likely requires a GPU cluster
* clone this repository


## Experiments

The recent version of `mml` (and its plugins) provides more comfort during reproduction :) Starting from version
`0.13.0` `mml` releases should be fully compatible (we recommend to use a `1.0.X` version).

* create a conda environment and install "recent" `mml`

```commandline
conda update -n base -c defaults conda  # OPTIONAL: update conda
conda create --name tf_recent  python=3.10 # create environment
conda activate tf_recent  # activate environment
pip install mml-core
```

* install plugins to provide bonus functionality

```commandline
# required for full reproduction
pip install mml-data              # provides the datasets
pip install mml-tags              # provides the shrinking tag
pip install mml-dimensionality    # compute dimensionality
pip install mml-similarity        # feature extraction
# optional for DKFZ infrastructure
pip install mml-drive             # faster data installation through network drive
pip install mml-lsf               # lsf cluster utils
```

* set env variables

```commandline
# create a mml.env file
mml-env-setup
# provide the env file location to conda environment as variable
pwd | conda env config vars set MML_ENV_PATH=$(</dev/stdin)/mml.env
# re-activate environmnet to have variable available
conda activate tf_recent
# edit env file
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
* depending on whether some of the plugin features shall be used there are additional variables (`drive` and `lsf`are
   not necessary)
    * if submitting to the LSF cluster is intended
      see [here](https://git.dkfz.de/imsy/ise/mml/-/tree/718acbbc9c1892a07bd1e8bdef6c6e34f20d3948/plugins/lsf)
    * if the network drive shall be used to speed up installation
      see [here](https://git.dkfz.de/imsy/ise/mml/-/tree/718acbbc9c1892a07bd1e8bdef6c6e34f20d3948/plugins/drive)

* finally this repo itself contains a plugin that provides evaluation capabilities

```commandline
# assume you cloned into this directory via "git clone ..."
cd task-fingerprinting/tf
pip install .
```

* test your installation by typing `mml --version` and check if all plugins are listed
* the plain `mml` command also checks the correct setting of basic env variables
* if running on the cluster you need to create this environments there as well, in addition set up a runner script
    * `mml.sh` that loads cuda, activates the `tf_recent` conda env and forwards the args to `mml` adding `sys=cluster`
* commands that need to be submitted (or run locally) are found in `notebooks/transfer_exps/recent_exps`
    * see `recent.ipynb` for details
    * you may submit job batches as e.g. `ssh user@host 'bash -s' < notebooks/transfer_exps/recent_exps/baseline.txt`

## Features and FIM extraction

This is straightforward with the recent `mml` version and described at the bottom of the `recent.ipynb` notebook.

## Task Transferability Evaluation

The downstream evaluation can be run from the same virtual environment. Except for the `tf` plugin no others are
required. By relying on the cached task distances (see `cache` folder) no feature extraction is necessary.




