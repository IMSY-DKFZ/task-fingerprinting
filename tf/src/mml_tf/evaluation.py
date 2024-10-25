from collections import Counter
from itertools import combinations
from typing import Dict, Sequence, Optional

import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from mml_tf.aggregate import AggregateStrategy, get_aggregated_raws, aggregate_observations
from mml_tf.distances import TaskDistances, get_closest, get_affinity_df
from mml_tf.experiments import SHRUNK, load_experiment, EXPERIMENTS
from mml_tf.tasks import shrink_map, get_valid_sources, shrinkable_tasks, train_tasks, test_tasks, task_infos


def calc_top_k(distances: TaskDistances, meta_metric: str, exp_results: pd.DataFrame, target_task: str, top_k: int,
               baseline: float, shrunk: bool, mode: str = 'avg', is_loss: bool = False, min_samples: int = 1) -> float:
    """
    Computes top-k suggestions performances for multiple meta_metrics.
     - regret has been suggested by https://arxiv.org/pdf/2010.06402.pdf
     - gain has been suggested by https://arxiv.org/pdf/1804.08328.pdf
     - ratio has been suggested as "Rel@1" by https://arxiv.org/pdf/2204.01403.pdf
     - delta
     - ranks

    :param distances: distances as calculated
    :param meta_metric: one of 'regret', 'gain', 'ratio', 'ranks', 'delta'
    :param exp_results: the actual measures performances, evaluated as .at[source, target]
    :param target_task: the target task
    :param shrunk: is the target task shrunk
    :param top_k: top k suggestions to consider as suggested by distances
    :param baseline: gives (shrunk) baseline performance for target tasks
    :param mode: may be 'avg' or 'best'
    :param is_loss: whether the underlying metric is the loss function
    :param min_samples: a minimum number of samples to qualify a task as potential source
    :return: meta metric performance for given measurements and distances
    """
    if mode not in ['avg', 'best']:
        raise ValueError('Invalid mode: {}'.format(mode))
    target_ref = shrink_map[target_task] if shrunk else target_task
    # selected top k source tasks as suggested by distances
    sources = get_closest(target_task=target_ref, distances=distances, budget=top_k, min_samples=min_samples)
    valid_sources = get_valid_sources(target_task, min_samples=min_samples)

    if meta_metric == 'rank':
        # ranks are sorted ascending between 0 and 1 (invert loss such that 1 is consistently best rank)
        ranks = exp_results[target_task].loc[valid_sources].rank(ascending=False if is_loss else True,
                                                                 method='average', pct=True).loc[sources]
        if mode == 'avg':
            return ranks.mean()
        else:
            return ranks.max()

    # compute oracle = best transfer performance
    if is_loss:
        oracle_transfer_score = exp_results[target_task].loc[valid_sources].min()
    else:
        oracle_transfer_score = exp_results[target_task].loc[valid_sources].max()
    top_k_performances = exp_results[target_task].loc[sources]
    # reduce to top performance within top k
    if mode == 'best':
        if is_loss:
            top_k_performances = [top_k_performances.min()]
        else:
            top_k_performances = [top_k_performances.max()]
    meta_values = []
    for performance in top_k_performances:
        if meta_metric == 'regret':
            source_gap = abs(oracle_transfer_score - performance)
            if is_loss:
                performance_gap = performance
            else:
                performance_gap = (1 - min(oracle_transfer_score, performance))
            if performance_gap == 0. == source_gap:
                meta_values.append(0.)
            else:
                meta_values.append(1 - (source_gap / performance_gap))  # we invert so regret means higher is better
        elif meta_metric == 'gain':
            if is_loss:
                meta_values.append(int(performance < baseline))
            else:
                meta_values.append(int(performance > baseline))
        elif meta_metric == 'ratio':
            if is_loss:
                meta_values.append(oracle_transfer_score / performance)
            else:
                meta_values.append(performance / oracle_transfer_score)
        elif meta_metric == 'delta':
            meta_values.append(performance - baseline)
    return np.mean(meta_values)


def calc_corr(distances: TaskDistances, meta_metric: str, exp_results: pd.DataFrame, target_task: str,
              shrunk: bool, is_loss: bool = False, min_samples: int = 1) -> float:
    """
    Computes correlation based meta metric.

     - weighedtau has been used by https://arxiv.org/pdf/2204.01403.pdf
     - variations of correlation meta metrics are common https://arxiv.org/pdf/2402.15231.pdf
     - limitations have been discussed by https://arxiv.org/pdf/2010.06402.pdf

    :param distances: distances as calculated
    :param meta_metric: one of 'weightedtau', 'pearson’, ‘kendall’, ‘spearman'
    :param exp_results: the actual measures performances, evaluated as .at[source, target]
    :param target_task: the target task
    :param shrunk: is the target task shrunk
    :param is_loss: whether the underlying metric is the loss function
    :param min_samples: a minimum number of samples to qualify a task as potential source
    :return: meta metric performance for given measurements and distances
    """
    affinities = get_affinity_df(distances=distances)
    target_ref = shrink_map[target_task] if shrunk else target_task
    valid_sources = get_valid_sources(target_task, min_samples=min_samples)
    measured = exp_results.loc[valid_sources][target_task]
    if is_loss:
        measured *= -1
    if meta_metric == 'weightedtau':
        method = lambda x, y: scipy.stats.weightedtau(x, y)[0]
    else:
        method = meta_metric
    return affinities[target_ref].loc[valid_sources].corr(measured, method=method)


def get_setup_stability_score(
        experiments_df,
        varying='target',
        fixing=('seed', 'metric', 'exp', 'meta metric')) -> float:
    """
    Get the setup stability score to the variation of a setup component.
    Suggested by https://arxiv.org/pdf/2204.01403.pdf

    :param experiments_df: Dataframe with columns as mentioned in varying and fixing, additionally 'distances' and 'score'
    :param varying: component to vary
    :param fixing: all other components to fix
    """
    setup_stability_scores = []
    # create the setups that share "fixing" column values
    for common_setup_values, multi_setup_results_df in experiments_df.groupby(list(fixing)):
        # extract the scores for each distances approach (in a sorted manner), iterating over values of "varying"
        score_vectors = []
        for varying_value, single_setup_results_df in multi_setup_results_df.groupby(varying):
            score_vectors.append(single_setup_results_df.set_index('distances').sort_index()['score'].values)
        # for each pair of experimental setups compute agreement of distances-ranking via weighted tau
        tau_values = []
        for vector_1, vector_2 in combinations(score_vectors, 2):
            tau_values.append(scipy.stats.kendalltau(vector_1, vector_2)[0])
        if np.isnan(tau_values).sum() == len(tau_values):
            print(f'common {common_setup_values} and varying {varying} is ONLY NAN VALUES')
        # average over all pairings
        setup_stability_scores.append(np.nanmean(tau_values))
    # average over all "common setups"
    return np.nanmean(setup_stability_scores)


def get_win_rates(experiments_df) -> Dict[str, float]:
    """
    Get the setup win rate for each distances within the experiments_df.
    Suggested by https://arxiv.org/pdf/2204.01403.pdf

    :param experiments_df: Dataframe with columns like, seed, exp, meta metric, etc., must have 'distances' and 'score'
    :return: dict with score for each distance approach
    """
    vary_setups = [col for col in experiments_df.columns if col not in ['distances', 'score']]
    counter = Counter()
    gbobj = experiments_df.groupby(vary_setups)
    for group_name, group_df in gbobj:
        _best_measures = group_df.set_index('distances')['score'].nlargest(n=1, keep='all').index.tolist()
        counter.update(_best_measures)
    return {dist: counter[dist] / len(gbobj) for dist in experiments_df['distances'].unique().tolist()}


def get_full_ranked_rates(experiments_df: pd.DataFrame, baseline_name: Optional[str] = None,
                          baseline_constant: Optional[float] = None) -> pd.DataFrame:
    """
    Get the setup ranking rate for each distances within the experiments_df.
    Inspired by https://arxiv.org/pdf/2204.01403.pdf in combination with Rankings Reloaded.

    :param experiments_df: Dataframe with columns like, seed, exp, meta metric, etc., must have 'distances' and 'score'
    :param baseline_name: (optional) name of a baseline to be added (e.g. no transfer or random transfer)
    :param baseline_constant: (optional) constant to be assigned to the baseline (e.g. 0. or 0.5)

    :return: dataframe with rank frequency for each distance approach (columns: distances, rank, fraction, count)
    """
    vary_setups = [col for col in experiments_df.columns if col not in ['distances', 'score']]
    all_distances = experiments_df['distances'].unique().tolist()
    if baseline_name:
        all_distances.append(baseline_name)
    all_counters = {dist: Counter() for dist in all_distances}
    # loop over all setups
    gbobj = experiments_df.groupby(vary_setups)
    for group_name, group_df in gbobj:
        sub_series = group_df.set_index('distances')['score']
        if baseline_name:
            sub_series = sub_series.copy()
            sub_series[baseline_name] = baseline_constant
        _rankings = sub_series.rank(method='min', ascending=False).to_dict()
        for k, v in _rankings.items():
            all_counters[k].update([v])
    # check
    all_totals = {dist: sum(counter.values()) for dist, counter in all_counters.items()}
    assert len(set(all_totals.values())) == 1
    assert len(gbobj) in set(all_totals.values())
    # build return df
    _row_data = []
    for dist in all_distances:
        for rank in range(1, len(all_distances) + 1):
            _row_data.append({'distances': dist, 'rank': rank, 'fraction': all_counters[dist][rank] / all_totals[dist],
                              'count': all_counters[dist][rank]})
    return pd.DataFrame(_row_data)


def get_evaluations(all_distances: Sequence[TaskDistances],
                    aggregates: Sequence[AggregateStrategy] = tuple([AggregateStrategy.MEAN]),
                    metrics: Sequence[str] = tuple(['AUROC', 'BA']),
                    experiments: Sequence[str] = tuple(EXPERIMENTS),
                    # task_list: Sequence[str] = tuple(target_tasks),
                    validation: bool = True,
                    corr_meta_metrics: Sequence[str] = tuple(['weightedtau']),
                    top_meta_metrics: Sequence[str] = tuple(['regret', 'rank']),
                    top_k: int = 3,
                    top_mode: str = 'avg',
                    disable_pbar: bool = False,
                    min_samples: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Evaluate a combination of setups.

    :param all_distances: distances to be evaluated
    :param aggregates: aggregation scheme(s) to consider
    :param metrics: metrics to consider
    :param experiments: experiments to consider
    :param validation: whether to evaluate on the validation tasks or on the training tasks
    :param corr_meta_metrics: meta metrics as accepted by calc_corr to consider
    :param top_meta_metrics: meta metrics as accepted by calc_top_k to consider
    :param top_k: top_k value for calc_top_k meta metrics
    :param top_mode: mode to consider for calc_top_k meta metrics
    :param disable_pbar: allows to disable progress bar updates
    :param min_samples: minimum number of samples to consider a task as source task
    :return: returns a dataframe with columns seed, exp, meta metric, distances, metric, score, target
    """
    # default values for min samples
    if min_samples is None:
        min_samples = {}
    for exp in experiments:
        if exp not in min_samples:
            min_samples[exp] = 1
    # set up task list
    task_list = test_tasks if validation else [t for t in train_tasks if task_infos.num_classes[t] < 40]
    # output collector
    evaluation_rows = []
    # iterations calculator
    total = len(aggregates) * len(metrics) * len(experiments) * len(all_distances) * len(task_list)
    # progressbar while looping
    with tqdm(disable=disable_pbar, total=total, desc='Calculating...') as bar:
        for aggregate in aggregates:
            for metric in metrics:
                is_loss = metric == 'loss'
                for experiment in experiments:
                    exp_df = aggregate_observations(
                        multi_seed_df=load_experiment(experiment_name=experiment, metric=metric, validation=validation),
                        strat=aggregate, is_loss=is_loss)
                    baselines_shrunk = get_aggregated_raws(strat=aggregate, metric=metric, shrunk=True,
                                                           validation=validation)
                    baselines_full = get_aggregated_raws(strat=aggregate, metric=metric, shrunk=False,
                                                         validation=validation)
                    for dist in all_distances:
                        for target in task_list:
                            if SHRUNK[experiment] and target in shrinkable_tasks:
                                baseline = baselines_shrunk[target]
                            else:
                                baseline = baselines_full[target]
                            for meta_metric in corr_meta_metrics:
                                score = calc_corr(distances=dist, meta_metric=meta_metric, exp_results=exp_df,
                                                  target_task=target, shrunk=SHRUNK[experiment], is_loss=is_loss,
                                                  min_samples=min_samples[experiment])
                                evaluation_rows.append(
                                    {'exp': experiment, 'distances': dist.name, 'metric': metric,
                                     'seed': aggregate.value, 'target': target, 'meta metric': meta_metric,
                                     'score': score})
                            for meta_metric in top_meta_metrics:
                                score = calc_top_k(distances=dist, meta_metric=meta_metric, exp_results=exp_df,
                                                   target_task=target, top_k=top_k, baseline=baseline,
                                                   shrunk=SHRUNK[experiment], mode=top_mode, is_loss=is_loss,
                                                   min_samples=min_samples[experiment])
                                evaluation_rows.append(
                                    {'exp': experiment, 'distances': dist.name, 'metric': metric,
                                     'seed': aggregate.value, 'target': target, 'meta metric': meta_metric,
                                     'score': score})
                            bar.update()
    return pd.DataFrame(evaluation_rows)
