import logging
import warnings
from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import List, Optional, Union, Callable, Sequence

import numpy as np
import pandas as pd
import scipy.stats
import torch.nn.functional

from mml_tf.aggregate import aggregate_observations, AggregateStrategy
from mml_tf.experiments import load_experiment
from mml_tf.paths import DATA_PATH, CACHE_PATH
from mml_tf.representations import (TaskRepresentations,
                                    TagBasedRepresentations,
                                    BinnedFeatureRepresentations,
                                    MeanAndCovarianceRepresentations,
                                    FisherEmbeddingRepresentations,
                                    AveragedFeatureRepresentations,
                                    DummyRepresentations, FullFeatureRepresentations)
from mml_tf.tasks import get_valid_sources, source_tasks, target_tasks, all_tasks_including_shrunk, all_tasks, \
    shrinkable_tasks, shrink_map, task_infos

# how to best transform task infos to be used in linear model
TRANSFORMS = ['boxcox', 'zscore']
transformed_task_infos = task_infos.get_transformed(transforms=TRANSFORMS)

# some display replacements, order is relevant for some legends
map_dist2printable = {
    'KLD-PP:NS-W:TS-100-BINS': 'bKLD(small,target)',
    'KLD-PP:NS-W:SN-1000-BINS': 'bKLD(large,source)',
    'KLD-PP:NS-1000-BINS': 'bKLD(large,unweighted)',
    'SEMANTIC': 'Manual',
    'FED': 'FED',
    'FID': 'FID',
    'P2L': 'P2L (w samples)',
    'KLD-PP:NN': 'P2L (w/o samples)',
    'VDNA-PP:NN-1000-BINS': 'VDNA',
    'no transfer': 'No transfer',  # no transfer is a dummy entry
}

# smallest probability to avoid zeros, corresponding to the default number of samples used (10_000) to determine bins
EPSILON = torch.tensor(1 / 10_000)


@dataclass
class HyperParameter:
    """
    Hyperparameters are a modification factor os task similarity by weighing in other meta-features of the source
    task. Inspired by https://arxiv.org/abs/1908.07630

    A higher factor means that tasks with a larger number of that factor will be preferred (e.g. sam=0.5 means that
    larger source tasks will be preferred. The meta-features are transformed in a preprocessing step that tries to
    unify the scale, hence equal weights will have roughly equal influence. See the `transformed_task_infos` attribute
    and the `get_affinity_df` function of this module.
    """
    sim: float = 1.0  # similarity (!) factor
    sam: float = 0.  # source sample factor
    cls: float = 0.  # source class factor
    dim: float = 0.  # source dim factor
    imb: float = 0.  # source imbalance factor


class TaskDistances:
    def __init__(self,
                 representations: TaskRepresentations,
                 name: str,
                 hp: Optional[HyperParameter] = None,
                 zscore_axis: Optional[int] = None,
                 cache: bool = True):
        """
        Most abstract class of task distances.

        :param TaskRepresentations representations: required representations (fingerprints)
        :param str name: identifier of the distances
        :param HyperParameter hp: optional HyperParameter to modify the task distances
        :param bool cache: whether to use the caching mechanism (encouraged for unique name)
        """
        self.rep = representations
        self.df = pd.DataFrame(columns=all_tasks_including_shrunk, index=all_tasks, dtype=float)
        self.name = name
        self.hp = hp
        self.cache = cache
        self.zscore_axis = zscore_axis
        self.df = self.load_distance()

    def load_distance(self) -> pd.DataFrame:
        """Load distances and prepare for usage."""
        df = self.calc()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df.isnull().sum().sum() > (len(shrinkable_tasks) + len(all_tasks)):
            warnings.warn(f'{self.name} has unexpected null values ({df.isnull().sum().sum()})')
        self._replace_shrunk_tasks(df)
        return self._transform_distance(df, self.zscore_axis)

    def calc(self) -> pd.DataFrame:
        """Calculate raw distances, will be cached."""
        # define cache path based on distance name
        df_path = CACHE_PATH / 'distances' / (self.name + '.csv')
        # if cached reuse
        if df_path.exists() and self.cache:
            return pd.read_csv(df_path, index_col=0, header=0)
        # not in cache means compute
        df = self._calc_impl()
        # cache
        if self.cache:
            df.to_csv(df_path)
        # and return
        return df

    def _calc_impl(self) -> pd.DataFrame:
        """Needs to be implemented by each inheriting TaskDistances and return the raw distance matrix"""
        raise NotImplementedError

    @staticmethod
    def _replace_shrunk_tasks(df) -> None:
        """Replaces new shrunk tagged tasks with the old version"""
        df.rename(index={t: shrink_map[t.split('+')[0]] for t in df.index if '+' in t},
                  columns={t: shrink_map[t.split('+')[0]] for t in df.index if '+' in t},
                  inplace=True)

    @staticmethod
    def _transform_distance(df, zscore_axis: Optional[int] = None) -> pd.DataFrame:
        """
        Simply scales each distance metric with zero mean and unit variance, also turning it into
        a similarity score (* -1).

        :param df: distance df
        :return: scaled distance df
        """
        _data = df.to_numpy().astype(float)
        _data[_data == 0] = np.nan  # ignore self-distance for zscoring
        _data = scipy.stats.zscore(_data, axis=zscore_axis, nan_policy='omit')
        new = pd.DataFrame(-1 * _data, columns=df.columns, index=df.index)
        return new


def get_affinity_df(distances: TaskDistances):
    """
    Calculate knowledge transfer affinity dataframe for a given distance measure with hyperparameters.

    :param distances: task distances instance
    :return: a dataframe where .at[source,target] describes transfer affinity (higher is better) from source to target
    """
    df = distances.df.copy().loc[list(source_tasks)]
    if distances.hp:
        df = df.apply(
            func=lambda x: distances.hp.sim * x
                           + distances.hp.sam * transformed_task_infos.num_samples[x.name]
                           + distances.hp.cls * transformed_task_infos.num_classes[x.name]
                           + distances.hp.dim * transformed_task_infos.dimensions[x.name]
                           + distances.hp.imb * transformed_task_infos.imbalance_ratios[x.name]
            , axis='columns')
    return df


def get_closest(target_task: str,
                distances: TaskDistances,
                budget: Optional[int] = None,
                min_samples: int = 1
                ) -> Union[str, List[str]]:
    """
    Function to calculate source task selection (respecting valid transfer and a computational budget).

    :param str target_task: the target task to request the source task(s) from
    :param TaskDistances distances: task distances
    :param Optional[int] budget: (optional) if given returns the top k choices instead of the single top choice
    :param Optional[int] min_samples: if given the returns are only from within source tasks of at least the given size
    :return: list of potential source tasks or a single task if budget is None
    """
    if '_shrink_train' in target_task:
        task_base = target_task[:target_task.find('_shrink_train')]
        real_target = task_base + ' --shrink_train ' + target_task.split('_')[-1]
    elif ' --shrink_train ' in target_task:
        task_base = target_task[:target_task.find(' --shrink_train ')]
        real_target = target_task
    else:
        task_base = target_task
        real_target = target_task
    # select target task column in df
    affinities = get_affinity_df(distances=distances)[real_target]
    # restrict to valid sources
    affinities = affinities.loc[get_valid_sources(task_base, min_samples=min_samples)]
    # indices for the semantically closest tasks
    suggestions = affinities.nlargest(n=1 if budget is None else budget, keep='all').index.tolist()
    # finally decide based on task size
    if budget is None:
        if len(suggestions) > 1:
            raise RuntimeError(f'{target_task=}, {budget=}, {distances.name}, {distances.hp=}, '
                               f'{distances.__class__}')
        return suggestions[0]
    else:
        if len(suggestions) != budget:
            if len(suggestions) > 1:
                raise RuntimeError(f'{target_task=}, {budget=}, {distances.name}, {distances.hp=}, '
                                   f'{distances.__class__}')
        return suggestions


def get_variety(distances: TaskDistances, shrunk: bool = False):
    """
    Calculates the variety of a source task selector. This means the size of actually chosen sources over the number of
    targets. A variety of one means every target has been assigned a distinct source.

    :return: float between zero and one, the larger, the more diverse are source selections
    """
    actual_targets = [shrink_map[t] for t in target_tasks] if shrunk else target_tasks
    chosen = {get_closest(target_task=t, distances=distances) for t in actual_targets}
    return len(chosen) / len(actual_targets)


##################################################
# below are implementations of various distances #
##################################################

class EnsembleDistances(TaskDistances):
    """Ensemble other task distances."""

    def __init__(self, base_dists: Sequence[TaskDistances], name: str, use_raws: bool = True,
                 weights: Optional[Sequence[float]] = None):
        assert len(base_dists) >= 2
        self.base_dists = base_dists
        self.use_raws = use_raws
        if weights is None:
            weights = [1.0] * len(base_dists)
        assert len(weights) == len(base_dists)
        self.weights = weights
        super().__init__(representations=DummyRepresentations(), name=name, cache=False)

    def _calc_impl(self) -> pd.DataFrame:
        if self.use_raws:
            all_dfs = [dist.calc() for dist in self.base_dists]
        else:
            # use transformed distances
            all_dfs = [-1 * dist.df for dist in self.base_dists]
        df = None
        for ix, other in enumerate(all_dfs):
            if df is None:
                df = self.weights[ix] * other
            else:
                df = df + (self.weights[ix] * other)
        return df


class SemanticDistances(TaskDistances):
    """Mimic manual source task selection by using semantic tags (and tiebreakers)."""

    def __init__(self, representations: TagBasedRepresentations):
        super().__init__(representations=representations, name='SEMANTIC')

    def _calc_impl(self) -> pd.DataFrame:
        df = self.df.copy()
        for s, t in combinations(all_tasks, 2):
            union = self.rep.mapping[s].union(self.rep.mapping[t])
            intersection = self.rep.mapping[s].intersection(self.rep.mapping[t])
            df.at[s, t] = 1 - (len(intersection) / len(union))
            df.at[t, s] = 1 - (len(intersection) / len(union))
            if t in shrinkable_tasks:
                df.at[s, shrink_map[t]] = 1 - (len(intersection) / len(union))
            if s in shrinkable_tasks:
                df.at[t, shrink_map[s]] = 1 - (len(intersection) / len(union))
        for task in df.index:
            df.at[task, task] = 0.0
        size_ranks = pd.Series(transformed_task_infos.num_samples).rank()
        dims_ranks = pd.Series(task_infos.dimensions).rank()
        for ix, t in enumerate(all_tasks):
            # slight uniqueness shift by source task size and data dimension
            df.loc[t] -= 0.001 * size_ranks[t]
            df.loc[t] -= 0.00001 * dims_ranks[t]
            df.loc[t] -= 0.0000001 * ix  # finally add some "random" selector
        return df


class OptimalDistances(TaskDistances):
    """Use the measured experimental results to come up with an optimal oracle"""

    def __init__(self, exp: str, agg: AggregateStrategy, metric: str):
        self.exp = exp
        self.agg = agg
        self.metric = metric
        super().__init__(representations=DummyRepresentations(), name=f'OPTIMAL-{exp}-{agg.value}-{metric}')

    def _calc_impl(self) -> pd.DataFrame:
        dev_df = aggregate_observations(
            multi_seed_df=load_experiment(experiment_name=self.exp, metric=self.metric, validation=False),
            strat=self.agg, is_loss=self.metric == 'loss')
        val_df = aggregate_observations(
            multi_seed_df=load_experiment(experiment_name=self.exp, metric=self.metric, validation=True),
            strat=self.agg, is_loss=self.metric == 'loss')
        columns = {}
        for target in self.rep.task_list:
            base_task = target.split(' --')[0]
            if base_task in val_df:
                columns[target] = val_df[base_task]
            elif base_task in dev_df:
                columns[target] = dev_df[base_task]
            else:
                # shrunk experiments sometimes have no measurements for targets with many classes
                columns[target] = None
        df = pd.DataFrame(columns)
        # transform to a "distance" (the lower the closer)
        if self.metric != 'loss':
            df *= -1
        return df


######################################
# variants of KLD for task distances #
######################################

class FeaturesDistances(TaskDistances):
    def __init__(self,
                 representations: Union[AveragedFeatureRepresentations, BinnedFeatureRepresentations,
                 MeanAndCovarianceRepresentations],
                 name: str,
                 is_symmetric: bool,
                 cache: bool = True):
        """
        Generic task distances for AverageFeatures and BinnedFeatures.

        :param representations: feature representations
        :param str name: distances name
        :param bool is_symmetric: indicator if distances are symmetric
        :param bool cache: whether to use the caching mechanism (encouraged for unique name)
        """
        self.is_symmetric = is_symmetric
        if isinstance(representations, MeanAndCovarianceRepresentations):
            assert name.startswith('FID')
        super().__init__(representations=representations, name=name, cache=cache)

    def _calc_impl(self) -> pd.DataFrame:
        """Generic computation scheme for features based task distances."""
        df = self.df.copy()
        assert any(isinstance(self.rep, _rep_cls) for _rep_cls in
                   [AveragedFeatureRepresentations, BinnedFeatureRepresentations, MeanAndCovarianceRepresentations])
        if not isinstance(self.rep, MeanAndCovarianceRepresentations):
            self.rep.to_cuda()
        for s, t in combinations(all_tasks, 2):
            df.at[s, t] = self._calc_single(s_task=s, t_task=t)
            if self.is_symmetric:
                # shortcut symmetric distances
                df.at[t, s] = df.at[s, t]
            else:
                df.at[t, s] = self._calc_single(s_task=t, t_task=s)
            if t in shrinkable_tasks:
                df.at[s, shrink_map[t]] = self._calc_single(s_task=s, t_task=shrink_map[t])
            if s in shrinkable_tasks:
                df.at[t, shrink_map[s]] = self._calc_single(s_task=t, t_task=shrink_map[s])
        for task in df.index:
            df.at[task, task] = 0.0
        if not isinstance(self.rep, MeanAndCovarianceRepresentations):
            self.rep.to_cpu()
        return df

    def _calc_single(self, s_task: str, t_task: str) -> float:
        """"""
        raise NotImplementedError


class PPandWeighFeaturesDistances(FeaturesDistances):
    """Capable of feature preprocessing and weighing."""

    def __init__(self,
                 representations: Union[AveragedFeatureRepresentations, BinnedFeatureRepresentations],
                 base_name: str,
                 is_symmetric: bool,
                 target_pp: str = 'norm',
                 source_pp: str = 'norm',
                 alpha: Optional[float] = None,
                 weighing_by: Optional[str] = None,
                 weights_rep: Optional[Union[AveragedFeatureRepresentations, torch.Tensor]] = None,
                 weights_pp: str = 'wo',
                 cache: bool = True,
                 seed: Optional[int] = None):
        """
        More advanced version of features distances - capable of preprocessing and weighing features.

        :param representations: feature representations
        :param base_name: base name of distances, will be modified based on settings
        :param is_symmetric: whether distances are symmetric
        :param target_pp: target features preprocessing (norm, soft, uniform or wo)
        :param source_pp: source features preprocessing (norm, soft, uniform or wo)
        :param alpha: alpha value for uniform smoothing
        :param weighing_by: whether weighing shall be done (source, target, both, provided) or not (None)
        :param weights_rep: either a provided fixed weighing (tensor) or average representations
        :param weights_pp: weight preprocessing (norm, soft, uniform or wo)
        :param cache: whether to use the caching mechanism (encouraged for unique name)
        :param seed: optional seed to be attached for resampling identifications
        """
        self.is_binned = isinstance(representations, BinnedFeatureRepresentations)
        # preprocessings
        self.alpha = alpha
        self.target_pp = self.get_pp(target_pp)
        self.source_pp = self.get_pp(source_pp)
        if target_pp == 'wo' == source_pp:
            # no preprocessing of features
            name = base_name
        else:
            name = base_name + '-PP:' + target_pp[0].upper() + source_pp[0].upper()
        if alpha and any(pp == 'uniform' for pp in [source_pp, target_pp]):
            name += f'-A:{alpha}'
        # weighing of features
        self.weighing_by = weighing_by
        self.weights = None
        if self.weighing_by is not None:
            if weighing_by == 'provided':
                assert isinstance(weights_rep, torch.Tensor)
                self.weights = weights_rep
                name += '-W:' + self.weighing_by[0].upper()
            else:
                assert weighing_by in ['source', 'target', 'both']
                assert isinstance(weights_rep, AveragedFeatureRepresentations)
                self.weights_pp = self.get_pp(weights_pp)
                self.weights = {t: self.weights_pp(weights_rep.mapping[t]).squeeze(0) for t in weights_rep.mapping}
                name += '-W:' + self.weighing_by[0].upper() + weights_pp[0].upper()
        if self.is_binned:
            name += f'-{representations.n_bins}-BINS'
        if seed is not None:
            name += f'-{representations.full_features.n_samples}-SAMPLES-{seed}-SEED'
        super().__init__(representations=representations, name=name, is_symmetric=is_symmetric, cache=cache)

    @staticmethod
    def norm_pp(t: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor by p=1 norm."""
        return torch.nn.functional.normalize(t, p=1, dim=1)

    @staticmethod
    def softmax_pp(t: torch.Tensor) -> torch.Tensor:
        """Preprocess a tensor by softmax."""
        return torch.nn.functional.softmax(t, dim=1)

    @staticmethod
    def no_pp(t: torch.Tensor) -> torch.Tensor:
        """Dummy identity preprocessing."""
        return t

    @staticmethod
    def uniform_pp(t: torch.Tensor, alpha: float) -> torch.Tensor:
        """Uniform smoothing preprocessing."""
        assert 0 < alpha < 1, f'invalid alpha for smoothing, was given {alpha}'
        return (alpha * torch.ones_like(t) * (1 / t.size(1))) + ((1 - alpha) * t)

    def get_pp(self, desc: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get feature preprocessing from description."""
        PP_MAP = {
            'wo': PPandWeighFeaturesDistances.no_pp,
            'norm': PPandWeighFeaturesDistances.norm_pp,
            'soft': PPandWeighFeaturesDistances.softmax_pp,
            'uniform': partial(PPandWeighFeaturesDistances.uniform_pp, alpha=self.alpha),
        }
        if desc not in PP_MAP:
            raise ValueError(f'{desc=} not recognized')
        else:
            return PP_MAP[desc]

    def get_weight(self, s_task: str, t_task: str) -> Union[torch.Tensor, None]:
        """Get the applicable weight."""
        if self.weighing_by:
            if self.weighing_by == 'provided':
                w = self.weights
            elif self.weighing_by == 'both':
                w = (self.weights[s_task] + self.weights[t_task]) / 2
            else:
                w_task = {'source': s_task, 'target': t_task}[self.weighing_by]
                w = self.weights[w_task]
        else:
            w = None
        return w


class KLDDistances(PPandWeighFeaturesDistances):
    """A variety of Kullback-Leibler divergence distances."""

    def __init__(self,
                 representations: Union[AveragedFeatureRepresentations, BinnedFeatureRepresentations],
                 target_pp: str = 'norm',
                 source_pp: str = 'soft',
                 alpha: Optional[float] = None,
                 weighing_by: Optional[str] = None,
                 weights_rep: Optional[Union[AveragedFeatureRepresentations, torch.Tensor]] = None,
                 weights_pp: str = 'wo',
                 clip: bool = False,
                 invert: bool = False,  # True means target -> q and source -> p in KLD calculation
                 cache: bool = True,
                 seed: Optional[int] = None):
        base_name = 'KLD'
        self.clip = clip
        if self.clip:
            base_name += '-C'
        self.invert = invert
        if self.invert:
            base_name += '-I'
        super().__init__(representations=representations, base_name=base_name, is_symmetric=False, target_pp=target_pp,
                         source_pp=source_pp, alpha=alpha, weighing_by=weighing_by, weights_rep=weights_rep,
                         weights_pp=weights_pp, cache=cache, seed=seed)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        p = self.target_pp(self.rep.mapping[t_task])
        q = self.source_pp(self.rep.mapping[s_task])
        w = self.get_weight(s_task=s_task, t_task=t_task)
        if self.invert:
            return self.my_kullback_leibler_divergence(p=q, q=p, clip=self.clip, feature_weights=w)
        return self.my_kullback_leibler_divergence(p=p, q=q, clip=self.clip, feature_weights=w)

    @staticmethod
    def my_kullback_leibler_divergence(p: torch.Tensor,
                                       q: torch.Tensor,
                                       clip: bool = False,
                                       feature_weights: Optional[torch.Tensor] = None) -> float:
        """Plain mathematical (batched) KL-divergence without any modification of probabilities."""
        if clip:
            p = p.clip(min=EPSILON.to(p.device))
            q = q.clip(min=EPSILON.to(p.device))
        binned = p.size(0) != 1
        n_features = p.size(0 if binned else 1)
        if feature_weights is None:
            feature_weights = torch.ones(n_features).to(p.device)
        else:
            feature_weights = feature_weights.to(p.device)
        if binned:
            # return (p / q).log().mul(p).nan_to_num().sum(dim=1).mul(feature_weights).sum().div(p.size(0)).item()
            return (p / q).log().mul(p).nan_to_num().sum(dim=1).mul(feature_weights).sum().div(
                feature_weights.sum()).item()
        else:
            # return (p / q).log().mul(p).nan_to_num().mul(feature_weights).sum().item()
            return (p / q).log().mul(p).nan_to_num().mul(feature_weights).sum().div(feature_weights.sum()).item()


class P2LDistances(KLDDistances):
    """As special variant of KLD distance that uses a hyperparameter modification. See https://arxiv.org/abs/1908.07630"""

    def __init__(self, representations: AveragedFeatureRepresentations):
        """Note that this is cached with the name KLD-PP:NN (!) - hyperparameters are disentangled from caching"""
        super().__init__(representations=representations, target_pp='norm', source_pp='norm', weighing_by=None,
                         weights_rep=None, weights_pp='wo', clip=False, cache=True, seed=None)
        self.hp = HyperParameter(sim=1.5, sam=1.)
        self.name = 'P2L'


class JSDistances(PPandWeighFeaturesDistances):
    """A variety of Jensen–Shannon divergence distances."""

    def __init__(self,
                 representations: Union[AveragedFeatureRepresentations, BinnedFeatureRepresentations],
                 alpha: Optional[float] = None,
                 weighing_by: Optional[str] = None,
                 weights_rep: Optional[Union[AveragedFeatureRepresentations, torch.Tensor]] = None,
                 weights_pp: str = 'wo',
                 clip: bool = False,
                 cache: bool = True,
                 seed: Optional[int] = None):
        base_name = 'JS'
        self.clip = clip
        if self.clip:
            base_name += '-C'
        super().__init__(representations=representations, base_name=base_name,
                         is_symmetric=False,  # only symmetric if weighing is non task dep.
                         target_pp='norm',
                         source_pp='norm', alpha=alpha, weighing_by=weighing_by, weights_rep=weights_rep,
                         weights_pp=weights_pp, cache=cache, seed=seed)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        p = self.target_pp(self.rep.mapping[t_task])
        q = self.source_pp(self.rep.mapping[s_task])
        w = self.get_weight(s_task=s_task, t_task=t_task)
        m = (p + q) / 2
        return (KLDDistances.my_kullback_leibler_divergence(p=p, q=m, clip=self.clip, feature_weights=w)
                + KLDDistances.my_kullback_leibler_divergence(p=q, q=m, clip=self.clip, feature_weights=w))


######################################
# variants of EMD for task distances #
######################################
class EMDDistances(PPandWeighFeaturesDistances):
    """Compute (weighted) EMD /VDNA from averaged/binned feature representations."""

    def __init__(self,
                 representations: Union[BinnedFeatureRepresentations, AveragedFeatureRepresentations],
                 soft_features: bool,
                 weighing_by: Optional[str] = None,
                 weights_rep: Optional[Union[AveragedFeatureRepresentations, torch.Tensor]] = None,
                 weights_pp: str = 'wo',
                 cache: bool = True):
        is_symmetric = weighing_by in [None, 'provided', 'both']
        source_pp = 'soft' if soft_features else 'norm'
        target_pp = 'soft' if soft_features else 'norm'
        if isinstance(representations, AveragedFeatureRepresentations):
            base_name = 'EMD-AVG'
        else:
            base_name = 'VDNA'
        super().__init__(representations=representations, base_name=base_name, is_symmetric=is_symmetric,
                         source_pp=source_pp, target_pp=target_pp, weighing_by=weighing_by, weights_rep=weights_rep,
                         weights_pp=weights_pp, cache=cache)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        diff = self.source_pp(self.rep.mapping[s_task]) - self.target_pp(self.rep.mapping[t_task])
        y = torch.cumsum(diff, dim=1)
        if self.weighing_by:
            w = self.get_weight(s_task=s_task, t_task=t_task).to(diff.device)
            return torch.cumsum(diff, dim=1).abs().sum(dim=1).mul(w).mean().item()
        else:
            return torch.mean(torch.sum(torch.abs(y), dim=1)).item()


######################################
# variants of COS for task distances #
######################################
class COSDistances(PPandWeighFeaturesDistances):
    """Compute cosine distance of features."""

    def __init__(self,
                 representations: Union[BinnedFeatureRepresentations, AveragedFeatureRepresentations],
                 soft_features: bool, seed: Optional[int] = None):
        source_pp = 'soft' if soft_features else 'norm'
        target_pp = 'soft' if soft_features else 'norm'
        assert isinstance(representations, AveragedFeatureRepresentations) or isinstance(representations,
                                                                                         BinnedFeatureRepresentations)
        self.n_features = representations.full_features.n_features
        super().__init__(representations=representations, is_symmetric=True, base_name='COS', source_pp=source_pp,
                         target_pp=target_pp, weighing_by=None, seed=seed)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        s = self.target_pp(self.rep.mapping[t_task])
        t = self.source_pp(self.rep.mapping[s_task])
        return torch.nn.functional.cosine_embedding_loss(s, t, torch.ones(self.n_features, device=s.device)).item()


#############################
# some other task distances #
#############################
class LNormDistances(PPandWeighFeaturesDistances):
    """Compute features distance with standard norms."""

    def __init__(self, representations: Union[AveragedFeatureRepresentations, BinnedFeatureRepresentations],
                 p: int,
                 soft_features: bool,
                 weighing_by: Optional[str] = None,
                 weights_rep: Optional[Union[AveragedFeatureRepresentations, torch.Tensor]] = None,
                 weights_pp: str = 'wo',
                 seed: Optional[int] = None,
                 cache: bool = True):
        base_name = f'L-{p}-NORM'
        self.p = p
        if isinstance(representations, AveragedFeatureRepresentations):
            source_pp = 'wo'
            target_pp = 'wo'
        else:
            source_pp = 'soft' if soft_features else 'norm'
            target_pp = 'soft' if soft_features else 'norm'
        is_symmetric = weighing_by in [None, 'provided', 'both']
        super().__init__(representations=representations, is_symmetric=is_symmetric, base_name=base_name,
                         source_pp=source_pp, target_pp=target_pp, weighing_by=weighing_by, weights_rep=weights_rep,
                         weights_pp=weights_pp, seed=seed, cache=cache)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        diff = self.source_pp(self.rep.mapping[s_task]) - self.target_pp(self.rep.mapping[t_task])
        if self.weighing_by:
            w = self.get_weight(s_task=s_task, t_task=t_task).to(diff.device)
            if diff.size(0) > 1:
                return diff.abs().pow(self.p).sum(dim=1).pow(1 / self.p).mul(w).mean().item()
            else:
                return diff.abs().mul(w).pow(self.p).sum(dim=1).pow(1 / self.p).mean().item()
        else:
            return diff.abs().pow(self.p).sum(dim=1).pow(1 / self.p).mean().item()


class LogDistances(FeaturesDistances):
    """Compute distance in log space."""

    def __init__(self, representations: AveragedFeatureRepresentations, w_s: int = 0, w_t: int = 0):
        self.w_s = w_s
        self.w_t = w_t
        super().__init__(representations=representations, name=f'LOG-S:{self.w_s}-T:{self.w_t}', is_symmetric=False,
                         cache=True)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        s = self.rep.mapping[s_task]
        t = self.rep.mapping[t_task]
        w_s = torch.tensor(self.w_s, device=s.device)
        w_t = torch.tensor(self.w_t, device=t.device)
        return (t / s).log().mul(t.pow(w_t)).mul(s.pow(w_s)).nan_to_num().sum().item()


class ExpDistances(FeaturesDistances):
    """Compute distance in exp space."""

    def __init__(self, representations: AveragedFeatureRepresentations, w_s: int = 0, w_t: int = 0):
        self.w_s = w_s
        self.w_t = w_t
        super().__init__(representations=representations, name=f'EXP-S:{self.w_s}-T:{self.w_t}', is_symmetric=False,
                         cache=True)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        s = self.rep.mapping[s_task]
        t = self.rep.mapping[t_task]
        w_s = torch.tensor(self.w_s, device=s.device)
        w_t = torch.tensor(self.w_t, device=t.device)
        return (t.exp() - s.exp()).mul(t.pow(w_t)).mul(s.pow(w_s)).nan_to_num().sum().item()


class FIDDistances(FeaturesDistances):
    """Fréchet Inception Distance"""

    def __init__(self, representations: MeanAndCovarianceRepresentations):
        super().__init__(representations=representations, name='FID', is_symmetric=True, cache=True)

    def _calc_single(self, s_task: str, t_task: str) -> float:
        return self.fid(self.rep.mapping[s_task], self.rep.mapping[t_task])

    @staticmethod
    def fid(mapping_a, mapping_b) -> float:
        """
        Code adapted from torch-fidelity.
        See https://github.com/toshas/torch-fidelity/blob/1e4eaa478fd42aeb31f8476ef7d5181ead9ead37/torch_fidelity/metric_fid.py
        """
        mu1 = mapping_a[0]
        mu2 = mapping_b[0]
        sigma1 = mapping_a[1]
        sigma2 = mapping_b[1]

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            logging.warning("fid calculation produces singular product; adding small epsilon to diagonal of cov "
                            "estimates")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # compute fid and return
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        return float(fid)


class GenericFEDDistances(TaskDistances):
    """Fisher Embedding Distances. See https://doi.org/10.1007/978-3-030-87202-1_42"""

    def __init__(self, representations: FisherEmbeddingRepresentations, layers: List[str], name: str):
        self.layers = layers
        super().__init__(representations=representations, name=name)

    def _calc_impl(self) -> pd.DataFrame:
        df = self.df.copy()
        embeddings = {task: torch.cat(
            tuple([tens.view(-1) for lay, tens in self.rep.mapping[task].items() if lay in self.layers])) for task in
            self.rep.task_list}
        for s, t in combinations(all_tasks, 2):
            df.at[s, t] = 1. - torch.nn.functional.cosine_similarity(embeddings[s], embeddings[t], dim=0).item()
            df.at[t, s] = df.at[s, t]  # fed is symmetric
            if t in shrinkable_tasks:
                df.at[s, shrink_map[t]] = 1 - torch.nn.functional.cosine_similarity(
                    embeddings[s], embeddings[shrink_map[t]], dim=0).item()
            if s in shrinkable_tasks:
                df.at[t, shrink_map[s]] = 1 - torch.nn.functional.cosine_similarity(
                    embeddings[t], embeddings[shrink_map[s]], dim=0).item()
        for task in df.index:
            df.at[task, task] = 0.0
        return df


class MMDDistances(TaskDistances):
    """Maximum Mean Discrepancy distance (using a cauchy kernel as explained in
    https://doi.org/10.1007/978-3-030-87202-1_42)."""

    def __init__(self, representations: FullFeatureRepresentations, blur: float = 0.05):
        self.blur = blur
        super().__init__(representations=representations, name=f'MMD-{representations.n_samples}-SAMPLES')

    def _calc_impl(self) -> pd.DataFrame:
        df = self.df.copy()
        reps = {task: torch.as_tensor(features).to(torch.device('cuda')) for task, features in self.rep.mapping.items()}
        for s, t in combinations(all_tasks, 2):
            df.at[s, t] = torch.mean(self.mmd_cauchy(reps[s], reps[t]).view(-1)).item()
            df.at[t, s] = df.at[s, t]  # mmd is symmetric
            if t in shrinkable_tasks:
                df.at[s, shrink_map[t]] = torch.mean(self.mmd_cauchy(reps[s], reps[shrink_map[t]]).view(-1)).item()
            if s in shrinkable_tasks:
                df.at[t, shrink_map[s]] = torch.mean(self.mmd_cauchy(reps[t], reps[shrink_map[s]]).view(-1)).item()
        for task in df.index:
            df.at[task, task] = 0.0
        return df

    def mmd_cauchy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Maximum Mean Discrepancy Cauchy Kernel computation. Code by Tim Adler (& IWR)."""
        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        XX = (self.blur * (self.blur + (rx.t() + rx - 2. * xx)) ** -1)
        YY = (self.blur * (self.blur + (ry.t() + ry - 2. * yy)) ** -1)
        XY = (self.blur * (self.blur + (rx.t() + ry - 2. * zz)) ** -1)
        return XX + YY - 2. * XY


class LoadMMLComputedDistances(TaskDistances):
    """Loads distances as precomputed by MML, must be placed as .csv files inside data/distances."""

    def __init__(self, file_name: str, hp: Optional[HyperParameter] = None):
        self.file_name = file_name
        super().__init__(representations=DummyRepresentations(), name=file_name.upper() + '-MML', hp=hp)

    def _calc_impl(self):
        return pd.read_csv(DATA_PATH / 'distances' / (self.file_name + '.csv'), header=0, index_col=0)


class LoadCachedDistances(TaskDistances):
    """Fast and convenient method to load any cached distances. Identify by file name inside cache/distances."""

    def __init__(self, name: str, hp: Optional[HyperParameter] = None, zscore_axis: Optional[int] = None):
        super().__init__(representations=DummyRepresentations(), name=name, hp=hp, zscore_axis=zscore_axis)

    def _calc_impl(self):
        raise RuntimeError(f'{self.name} is not cached yet!')
