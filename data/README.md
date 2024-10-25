# data folder

This folder may hold extracted task data. It is structured as follows:
 * `features` contains extracted features for each task
 * `fims` similarly holds extracted fisher information matrices for each task
 * `results-imagenet-...` corresponds to [`timm`](https://github.com/huggingface/pytorch-image-models/commits/main/results/results-imagenet.csv) imagenet results at various commits

Note that the intermediate task similarities results are provided within the `cache` folder, and it is not necessary to 
extract features and fims to run inference notebooks. The extracted features and fims are not provided alongside.

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

