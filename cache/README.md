# Cache folder

This folder caches the results of `notebooks/0_fill_cache.ipynb` as intermediate results for the community and faster 
usage of other notebooks. Next it caches some task information in `task_infos.csv` that are loaded automatically while 
importing `mml_tf.tasks`. Furthermore, it caches results from `notebooks/6_vary_hyperparameters.ipynb` (`dim_df.csv` 
and `sam_df.csv`). The former caches are auto-used, so if you want to recompute, just delete the content in 
`cache/distances` and/or `cache/exp`.

## distances

Estimated task distances as CSV files (the lower the distance, the higher the assumed transferability). Note that these 
cached distances can be loaded fast via `mml_tf.distances.LoadCachedDistances`. For details on the caching see the 
`mml_tf.distances.TaskDistances` base class inside the `tf/src` directory. A CSV file is oriented as source task as 
row names and target tasks as column names. The name of the files represent a unique combination of kwargs for one of 
the TaskDistances. See `mml_tf.distances` to identify the TaskDistances corresponding to each file.

## exp

Measured task transferability success. These are the extracted results from the transfer experiments that are compiled 
using the `mml_tf.experiments.load_experiment` convenience wrapper. This wrapper simplifies the access by 
unification across experiments and auto-cache usage. The file pattern is as follows:
`load_EXPID_experiment%METRIC%SHRUNK%VALIDATION`, where
 * `EXPID` is either `arch`, `pretrain`, `augmentation` or `multi_task` for the four transfer scenarios or `baseline` for the single task baseline
 * `METRIC` is either `AUROC` or `BA` and describes the performance measure used to evaluate the resulting model on the target task test subset
 * `SHRUNK` is either `True` or `False` and indicates if the experiment was performed with a "shrunk" target task
 * `VALIDATION` is either `True` or `False` and indicates if the development or validation tasks are used

The CSV files have the following columns:
 * a row identifier (no header name of the column)
 * `r` the repetition number (random seed of the exp)
 * `s` the source task
 * `t` the target task
 * `p` the performance metric evaluation

The baseline CSV files (`load_baseline_experiment%....csv`) are slightly different (as there are no source tasks), they 
have column names corresponding to the target task and 3 rows for the three repetitions and entries according to the 
performance metric.

## hyperparameter modifications

The two files `dim_df.csv` and `sam_df.csv` are the evaluation results of modifications on task distances. One is based 
on the number of SAMples in the source task, the other is based on an estimated DIMension of the source task. See 
`notebooks/6_vary_hyperparameters.ipynb` as well as `mml_tf.distances.HyperParameter` for more details.
