import functools
import os
from pathlib import Path
from typing import Callable

import mml.interactive
import numpy as np
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from mml_tf.paths import MML_PROJECTS_VALIDATION, CACHE_PATH, MML_PROJECTS_TRAIN
from mml_tf.tasks import test_tasks, shrinkable_tasks, non_shrinkable_tasks, get_valid_sources, target_tasks, \
    task_infos, all_tasks, train_tasks

# stores gpu hours globally
GPU_TIME = {}
RERUNS = 3  # number of random seeds used
# keep order but can rename
EXPERIMENTS = ['Model<br>Architecture', 'Pretraining<br>Data', 'Augmentation<br>Policy', 'Co-Training<br>Data']
# can only choose freely for first and last exp
SHRUNK = {'Model<br>Architecture': True, 'Pretraining<br>Data': True, 'Augmentation<br>Policy': True,
          'Co-Training<br>Data': True}
# may also use others, e.g. Accuracy, loss, etc.
METRICS = ['BA', 'AUROC']


def get_performances(loaded_models, metric='AUROC', expected_reruns=1, gpu_time_indicator='other',
                     select_subdict_fct=None, expected_tasks=target_tasks, expected_subs=None):
    """Generic extraction function for performances of loaded models."""
    print('Extracting ...')
    # match metric names to torchmetrics
    if metric == 'BA':
        metric = 'Recall'
    metric = 'val/' + metric
    # For each target task, what is the average performance of the models
    performances = {}
    gpu_counter = 0.0
    none_counter = 0
    for task, task_models in loaded_models.items():
        if '_shrink_train' in task:
            task = task[:task.find('_shrink_train')]
        performances[task] = {}
        for model in task_models:
            # extract the source and metric from config, this varies between transfer scenarios
            if select_subdict_fct:
                adapted_path = Path(os.getenv('MML_RESULTS_PATH')) / model.pipeline.relative_to(
                    os.getenv('MML_CLUSTER_RESULTS_PATH')) if str(model.pipeline).startswith(
                    os.getenv('MML_CLUSTER_RESULTS_PATH')) else model.pipeline
                pipeline = OmegaConf.load(adapted_path)
                sub, sub_metric = select_subdict_fct(pipeline, metric, task)
            else:
                sub, sub_metric = 'default', metric
            if sub not in performances[task]:
                performances[task][sub] = []
            # in some versions validation is logged before training metrics, hence we fall back to the second to last
            index = -1 if sub_metric in model.metrics[-1] else -2
            try:
                value = model.metrics[index][sub_metric]
            except KeyError:
                print(f'{model.pipeline=}')
                raise
            if value is None:
                print(f'found None value at {model.pipeline}')
                none_counter += 1
            performances[task][sub].append(value)
            gpu_counter += model.training_time
    for task in loaded_models:
        # handle shrunk tasks gracefully
        if '_shrink_train' in task:
            task = task[:task.find('_shrink_train')]
        # perform checks to ensure all expected data behaves well
        for sub in performances[task]:
            if len(performances[task][sub]) != expected_reruns:
                print(
                    f'Incorrect number of performance for task {task} and sub {sub}, have {len(performances[task][sub])}')
            if any([perf == float('inf') for perf in performances[task][sub]]):
                print(f'Invalid INF performance for task {task} and sub {sub}, have {performances[task][sub]}')
            if any([perf is None or np.isnan(perf) for perf in performances[task][sub]]):
                print(f'Invalid None / NAN performance for task {task} and sub {sub}, have {performances[task][sub]}')
    for t in expected_tasks:
        # additional check
        if t not in performances:
            print(f'Missing task {t}!')
    if expected_subs:
        # additional check
        for t in performances:
            if set(performances[t].keys()) != set(expected_subs):
                print(f'Sub mismatch for task {t}: got {performances[t].keys()}')
    if gpu_time_indicator:
        # only update if not yet provided
        if gpu_time_indicator not in GPU_TIME:
            GPU_TIME[gpu_time_indicator] = gpu_counter
            print(f'Total GPU time for {gpu_time_indicator} was {gpu_counter}s.')
    if none_counter > 0:
        print(f'Total on {none_counter} invalid entries found')
    return performances


def cache_exp_df(_callable: Callable):
    """Decorator to cache experimental results after extraction automatically."""

    @functools.wraps(_callable)
    def wrapper(*args, **kwargs):
        # ensure kwarg usage
        if len(args) != 0:
            raise ValueError('please use cached loaders only with keyword arguments')
        # sort kwargs and create deterministic kwarg dependent
        vals = [str(v) for k, v in sorted(kwargs.items(), key=lambda item: item[0])]
        df_path = CACHE_PATH / 'exp' / ('%'.join([_callable.__name__] + [v.replace('/', '%') for v in vals]) + '.csv')
        # if cached reuse
        if df_path.exists():
            return pd.read_csv(df_path, index_col=0)
        # not in cache means compute
        df = _callable(**kwargs)
        # cache
        df.to_csv(df_path)
        # and return
        return df

    return wrapper


@cache_exp_df
def load_baseline_experiment(metric: str, shrunk: bool, validation: bool):
    project_name = MML_PROJECTS_VALIDATION['shrunk_baseline'] if shrunk else MML_PROJECTS_VALIDATION['full_baseline']
    models_list = []
    print('Loading ...')
    for ix in range(RERUNS):
        models_list.append(mml.interactive.load_project_models(f'{project_name}_{ix}'))
    merged_models = mml.interactive.merge_project_models(models_list)

    task_pool = test_tasks if validation else train_tasks
    expected = [t for t in task_pool if t in shrinkable_tasks] if shrunk else task_pool

    performances = get_performances(loaded_models=merged_models, metric=metric, expected_reruns=RERUNS,
                                    gpu_time_indicator=f"baseline_{'shrunk' if shrunk else 'full'}",
                                    select_subdict_fct=None, expected_tasks=expected, expected_subs=None)
    # rearrange
    _perf = {task: performances[task]['default'] for task in expected}
    df = pd.DataFrame(_perf)
    return df


#################################
# EXPERIMENT 1: MODEL TRANSFER  #
#################################
model_transfer_arch_list = ['tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns',
                            'tf_efficientnet_cc_b0_4e', 'swsl_resnet50', 'ssl_resnext50_32x4d', 'regnetx_032',
                            'regnety_032', 'rexnet_100', 'ecaresnet50d', 'cspdarknet53', 'mixnet_l', 'cspresnext50',
                            'cspresnet50', 'ese_vovnet39b', 'resnest50d', 'hrnet_w18', 'skresnet34',
                            'mobilenetv3_large_100', 'res2net50_26w_4s'
                            ]


@cache_exp_df
def load_arch_experiment(metric: str, shrunk: bool = False, validation: bool = True) -> pd.DataFrame:
    MML_PROJECTS = MML_PROJECTS_VALIDATION if validation else MML_PROJECTS_TRAIN
    project_name = MML_PROJECTS['arch_search']
    shrunk_proj_name = MML_PROJECTS['arch_shrunk']
    models_list = []
    print('Loading ...')
    for ix in range(RERUNS):
        models_list.append(mml.interactive.load_project_models(f'{project_name}_{ix}'))
    merged_models = mml.interactive.merge_project_models(models_list)
    shrunk_models_list = list()
    if shrunk:
        print('Loading shrunk ...')
        for ix in range(RERUNS):
            shrunk_models_list.append(mml.interactive.load_project_models(f'{shrunk_proj_name}_{ix}'))
    merged_shrunk_models = mml.interactive.merge_project_models(shrunk_models_list)

    def arch_select_fct(pipeline: DictConfig, metric, task):
        return pipeline.arch.classification.id, metric

    performances = get_performances(loaded_models=merged_models, metric=metric, expected_reruns=RERUNS,
                                    gpu_time_indicator='full_arch_val' if validation else 'full_arch_dev',
                                    select_subdict_fct=arch_select_fct,
                                    expected_tasks=all_tasks if validation else train_tasks,
                                    expected_subs=model_transfer_arch_list)
    if shrunk:
        task_pool = test_tasks if validation else train_tasks
        target_performances = get_performances(loaded_models=merged_shrunk_models, metric=metric,
                                               expected_reruns=RERUNS,
                                               gpu_time_indicator='shrunk_arch_val' if validation else 'shrunk_arch_dev',
                                               select_subdict_fct=arch_select_fct,
                                               expected_tasks=[t for t in task_pool if t in shrinkable_tasks],
                                               expected_subs=model_transfer_arch_list)
        # have not repeated these experiments, just copying values there
        for s in task_pool:
            if s in non_shrinkable_tasks:
                target_performances[s] = performances[s]
    else:
        target_performances = performances
    # rearrange
    df_collection = []
    arch_report = {arch: 0 for arch in model_transfer_arch_list}
    for repetition in range(RERUNS):
        rep_src_perfs = {_source: {k: v[repetition] for k, v in _dict.items()} for _source, _dict in
                         performances.items()}
        for s in all_tasks:
            if s in test_tasks and validation is False:
                continue
            best_arch = pd.Series(rep_src_perfs[s]).idxmax()
            arch_report[best_arch] += 1
            for t in all_tasks:
                if (t in test_tasks and validation is False) or (t in train_tasks and validation is True):
                    continue
                if s in get_valid_sources(t):
                    df_collection.append(
                        {'r': repetition, 's': s, 't': t, 'p': target_performances[t][best_arch][repetition]})
    df = pd.DataFrame(df_collection)
    print(f'{metric=} {arch_report=}')
    return df


####################################
# EXPERIMENT 2: TRANSFER LEARNING  #
####################################
@cache_exp_df
def load_pretrain_experiment(metric: str, validation: bool = True) -> pd.DataFrame:
    MML_PROJECTS = MML_PROJECTS_VALIDATION if validation else MML_PROJECTS_TRAIN
    project_name = MML_PROJECTS['transfer']
    models_list = []
    print('Loading ...')
    for ix in range(RERUNS):
        models_list.append(mml.interactive.load_project_models(f'{project_name}_{ix}'))
    merged_models = mml.interactive.merge_project_models(models_list)

    def trans_select_fct(pipeline: DictConfig, metric, task):
        return pipeline.mode.pretrain_task, metric

    task_pool = test_tasks if validation else train_tasks
    performances = get_performances(loaded_models=merged_models, metric=metric, expected_reruns=RERUNS,
                                    gpu_time_indicator='pretraining_val' if validation else 'pretraining_dev',
                                    select_subdict_fct=trans_select_fct,
                                    expected_tasks=[t for t in task_pool if task_infos.num_classes[t] < 40],
                                    expected_subs=None)
    # rearrange
    df_collection = []
    for repetition in range(RERUNS):
        for t in all_tasks:
            if ((t in test_tasks and validation is False)
                    or (t in train_tasks and validation is True)
                    or (task_infos.num_classes[t] >= 40)):
                continue
            for s in get_valid_sources(t):
                df_collection.append({'r': repetition, 's': s, 't': t, 'p': performances[t][s][repetition]})
    df = pd.DataFrame(df_collection)
    return df


######################################
# EXPERIMENT 3: AUG POLICY TRANSFER  #
######################################
@cache_exp_df
def load_augmentation_experiment(metric: str, validation: bool = True) -> pd.DataFrame:
    MML_PROJECTS = MML_PROJECTS_VALIDATION if validation else MML_PROJECTS_TRAIN
    project_name = MML_PROJECTS['aa_infer']
    models_list = []
    print('Loading ...')
    for ix in range(RERUNS):
        models_list.append(mml.interactive.load_project_models(f'{project_name}_{ix}'))
    merged_models = mml.interactive.merge_project_models(models_list)

    def policy_select_fct(pipeline: DictConfig, metric, task):
        return pipeline.augmentations.source, metric

    task_pool = test_tasks if validation else train_tasks
    performances = get_performances(loaded_models=merged_models, metric=metric, expected_reruns=RERUNS,
                                    gpu_time_indicator=f'augmentations_val' if validation else 'augmentations_dev',
                                    select_subdict_fct=policy_select_fct,
                                    expected_tasks=[t for t in task_pool if task_infos.num_classes[t] < 40],
                                    expected_subs=None)
    # rearrange
    df_collection = []
    for repetition in range(RERUNS):
        for t in all_tasks:
            if ((t in test_tasks and validation is False)
                    or (t in train_tasks and validation is True)
                    or (task_infos.num_classes[t] >= 40)):
                continue
            for s in get_valid_sources(t):
                df_collection.append({'r': repetition, 's': s, 't': t, 'p': performances[t][s][repetition]})
    df = pd.DataFrame(df_collection)
    return df


######################################
# EXPERIMENT 4: MULTI-TASK LEARNING  #
######################################
@cache_exp_df
def load_multi_task_experiment(metric: str, shrunk: bool = False, validation: bool = True) -> pd.DataFrame:
    MML_PROJECTS = MML_PROJECTS_VALIDATION if validation else MML_PROJECTS_TRAIN
    project_name = MML_PROJECTS['multi_task']
    shrunk_proj_name = MML_PROJECTS['multi_shrunk']
    models_list = []
    print('Loading ...')
    for ix in range(RERUNS):
        models_list.append(mml.interactive.load_project_models(f'{project_name}_{ix}'))
    merged_models = mml.interactive.merge_project_models(models_list)
    shrunk_models_list = list()
    if shrunk:
        print('Loading shrunk ...')
        for ix in range(RERUNS):
            shrunk_models_list.append(mml.interactive.load_project_models(f'{shrunk_proj_name}_{ix}'))
    merged_shrunk_models = mml.interactive.merge_project_models(shrunk_models_list)

    def multi_select_fct(pipeline: DictConfig, metric, task):
        met = f'{metric.split("/")[0]}/{task}/{metric.split("/")[1]}'
        if metric.split('/')[1] == 'loss':
            raise ValueError('loss not compatible with multi task learning evaluation')
        return pipeline.mode.possible_tasks[0], met

    performances = get_performances(loaded_models=merged_models, metric=metric, expected_reruns=RERUNS,
                                    gpu_time_indicator='full_multi_val' if validation else 'full_multi_dev',
                                    select_subdict_fct=multi_select_fct,
                                    expected_tasks=test_tasks if validation else train_tasks,
                                    expected_subs=None)

    if shrunk:
        def multi_shrunk_select_fct(pipeline: DictConfig, metric, task):
            met = f'{metric.split("/")[0]}/{task} --shrink_train 800/{metric.split("/")[1]}'
            if metric.split('/')[1] == 'loss':
                raise ValueError('loss not compatible with multi task learning evaluation')
            return pipeline.mode.possible_tasks[0], met

        task_pool = test_tasks if validation else train_tasks
        target_performances = get_performances(loaded_models=merged_shrunk_models, metric=metric,
                                               expected_reruns=RERUNS,
                                               gpu_time_indicator='shrunk_multi_val' if validation else 'shrunk_multi_dev',
                                               select_subdict_fct=multi_shrunk_select_fct,
                                               expected_tasks=[t for t in task_pool if t in shrinkable_tasks],
                                               expected_subs=None)
        # have not repeated these experiments, just copying values there
        for t in task_pool:
            if t in non_shrinkable_tasks:
                target_performances[t] = performances[t]
    else:
        target_performances = performances
    # rearrange
    df_collection = []
    for repetition in range(RERUNS):
        for t in all_tasks:
            if (t in test_tasks and validation is False) or (t in train_tasks and validation is True):
                continue
            for s in get_valid_sources(t):
                df_collection.append(
                    {'r': repetition, 's': s, 't': t, 'p': target_performances[t][s][repetition]})
    df = pd.DataFrame(df_collection)
    return df


def load_experiment(experiment_name: str, metric: str, validation: bool = True) -> pd.DataFrame:
    """Convenience wrapper around the individual loading functions"""
    loader_map = {exp: func for exp, func in zip(EXPERIMENTS, [load_arch_experiment, load_pretrain_experiment,
                                                               load_augmentation_experiment,
                                                               load_multi_task_experiment])}
    if EXPERIMENTS.index(experiment_name) in [0, 3]:
        return loader_map[experiment_name](metric=metric, shrunk=SHRUNK[experiment_name], validation=validation)
    else:
        return loader_map[experiment_name](metric=metric, validation=validation)
