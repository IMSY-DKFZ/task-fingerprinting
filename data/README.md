# data folder

This folder may hold extracted task data. It is structured as follows:
 * `auto_augmentations` contains the (externally) trained policies from FasterAutoAugment
 * `features` contains extracted features for each task
 * `fims` similarly holds extracted fisher information matrices for each task
 * `nnssl_results` holds the output of the feature extractions on the Medical Segmentation Decathlon tasks
 * `results-imagenet-...` corresponds to [`timm`](https://github.com/huggingface/pytorch-image-models/commits/main/results/results-imagenet.csv) imagenet results at various commits

Note that the intermediate task similarities results are provided within the `cache` folder, and it is not necessary to 
extract features and fims to run inference notebooks. The extracted features and fims are not provided alongside.

## auto_augmentations

These are injected to be loadable in the script `tf/src/mml_tf/activate.py`, which is automatically executed as part of 
plugin loading when `mml-tf` is installed and `mml` is started. The complicated way for this integration is a major 
refactoring of `mml` prior to open sourcing.

## features

After running the feature extraction command from the `transfer_exps/recent_exps/recent.ipynb` notebook 
place the content of the `FEATURES` sub-folder within the `pami2_features` project of your `MML_RESULTS_PATH` inside. 
It should look somehow as follows:

```commandline
.
├── aptos19_blindness_detection
│   └── features_0001.npy
├── aptos19_blindness_detection+shrink_train?800
│   └── features_0001.npy
├── barretts_esophagus_diagnosis
│   └── features_0001.npy
├── bean_plant_disease_classification
│   └── features_0001.npy
├── bean_plant_disease_classification+shrink_train?800
...
```

> **Note**: The `clip_features`, `dino_features`, and `mae_features` folders are the corresponding extracted features 
> folders for the additional backbones tested for the generalization experiments.

## fims
After running the fim extraction command from the `transfer_exps/recent_exps/recent.ipynb` notebook 
place the content of the `FIMS` sub-folder within the `pami2_fims_recent` project of your `MML_RESULTS_PATH` inside. 
It should look somehow as follows:

```commandline
.
├── aptos19_blindness_detection
│   └── fim_0001.pkl
├── aptos19_blindness_detection+shrink_train?800
│   └── fim_0001.pkl
├── barretts_esophagus_diagnosis
│   └── fim_0001.pkl
├── bean_plant_disease_classification
│   └── fim_0001.pkl
├── bean_plant_disease_classification+shrink_train?800
...
```

## nnssl_results
Results folder based on [nnssl](https://github.com/MIC-DKFZ/nnssl) on the tasks from the 
[Medical Segmentation Decathlon](http://medicaldecathlon.com/). The folder looks roughly as follows:

```commandline
.
├── Dataset001_BrainTumour
│   └── BaseMAETrainer_feature__nnsslPlans__onemmiso
│     └── fold_all
│       └── img_log
│         └── feat
│           └── BrainTumour__BrainTumour__BRATS_001__unknown_session__0__BRATS_001_0001_0.npz
│           └── BrainTumour__BrainTumour__BRATS_001__unknown_session__0__BRATS_001_0001_1.npz
│           └── BrainTumour__BrainTumour__BRATS_001__unknown_session__0__BRATS_001_0001_2.npz
│           └── BrainTumour__BrainTumour__BRATS_001__unknown_session__1__BRATS_001_0002_0.npz
│           ...
├── Dataset002_Heart
│   └── ...
├── Dataset003_Liver
│   └── ...
├── Dataset004_Hippocampus
│   └── ...
├── Dataset005_Prostate
...
```
