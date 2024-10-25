from enum import Enum
from typing import Dict

import pandas as pd

from mml_tf.experiments import load_baseline_experiment


class AggregateStrategy(Enum):
    MEAN = 'mean'
    WORST = 'min'
    BEST = 'max'
    MEDIAN = 'median'
    FIRST = 'first'
    SECOND = 'second'
    THIRD = 'third'


def aggregate_observations(multi_seed_df: pd.DataFrame, strat: AggregateStrategy,
                           is_loss: bool = False) -> pd.DataFrame:
    """Turns multi metric observations as loaded with multiple random seeds into a single source-target dataframe"""
    if strat in [AggregateStrategy.FIRST, AggregateStrategy.SECOND, AggregateStrategy.THIRD]:
        keep_idx = {AggregateStrategy.FIRST: 0, AggregateStrategy.SECOND: 1, AggregateStrategy.THIRD: 2}[strat]
        filtered = multi_seed_df[multi_seed_df['r'] == keep_idx]
    else:
        filtered = multi_seed_df
    # step 2: regroup
    regrouped = filtered.drop(['r'], axis='columns').groupby(by=['s', 't'])
    # step 3: apply strategy
    if strat == AggregateStrategy.MEAN:
        applied = regrouped.mean()
    elif (strat == AggregateStrategy.WORST and not is_loss) or (is_loss and strat == AggregateStrategy.BEST):
        applied = regrouped.min()
    elif strat == AggregateStrategy.MEDIAN:
        applied = regrouped.median()
    elif (strat == AggregateStrategy.BEST and not is_loss) or (is_loss and strat == AggregateStrategy.WORST):
        applied = regrouped.max()
    else:
        applied = regrouped.max()  # reduce for strategies first, second and third
    # step 4: reformat
    reformatted = applied.unstack().droplevel(level=0, axis=1)
    # one may now access reformatted.at[source, target]
    return reformatted


def get_aggregated_raws(strat: AggregateStrategy, shrunk: bool, metric: str, validation: bool) -> Dict[str, float]:
    """Turns multi metric observations as loaded with multiple random seeds into a single source-target dataframe"""
    all_reps = load_baseline_experiment(metric=metric, shrunk=shrunk, validation=validation)  # baselines exps
    is_loss = metric == 'loss'
    if strat == AggregateStrategy.MEAN:
        reduced = all_reps.mean()
    elif (strat == AggregateStrategy.WORST and not is_loss) or (is_loss and strat == AggregateStrategy.BEST):
        reduced = all_reps.min()
    elif (strat == AggregateStrategy.BEST and not is_loss) or (is_loss and strat == AggregateStrategy.WORST):
        reduced = all_reps.max()
    elif strat == AggregateStrategy.MEDIAN:
        reduced = all_reps.median()
    elif strat == AggregateStrategy.FIRST:
        reduced = all_reps.loc[0]
    elif strat == AggregateStrategy.SECOND:
        reduced = all_reps.loc[1]
    elif strat == AggregateStrategy.THIRD:
        reduced = all_reps.loc[2]
    else:
        raise ValueError(f'given {strat=} not valid')
    return reduced.to_dict()
