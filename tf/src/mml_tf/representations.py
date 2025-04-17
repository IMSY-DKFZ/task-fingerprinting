from typing import List, Dict, Optional, Union, Tuple, Set

import mml.interactive
import numpy as np
import torch
from rich.progress import track

from mml_tf.paths import DATA_PATH
from mml_tf.tasks import all_tasks_including_shrunk, new_to_old, train_tasks, task_infos, old_to_new

Representation = Optional[Union[Set[str], torch.Tensor, np.ndarray, Tuple[np.ndarray, np.ndarray]]]


class TaskRepresentations:
    def __init__(self, name: str, task_list: List[str] = all_tasks_including_shrunk):
        self.task_list = task_list
        self.name = name
        self.mapping: Dict[str, Representation] = {t: None for t in self.task_list}

    def load_representations(self):
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return all(v is not None for v in self.mapping.values())


class DummyRepresentations(TaskRepresentations):
    def __init__(self):
        super().__init__('dummy')


class FullFeatureRepresentations(TaskRepresentations):
    """Numpy arrays like (samples x features)"""

    def __init__(self, task_list: List[str] = all_tasks_including_shrunk):
        super().__init__(task_list=task_list, name='Full Features')

    def load_representations(self):
        features_path = DATA_PATH / 'features'
        for task in features_path.iterdir():
            if '+' in task.name:
                _task = task.name.replace('+shrink_train?800', ' --shrink_train 800')
            else:
                _task = task.name
            assert _task in self.task_list, 'invalid task name'
            self.mapping[_task] = np.load(features_path / task.name / 'features_0001.npy')
        assert self.is_loaded(), 'features missing'

    @property
    def n_features(self) -> int:
        assert self.is_loaded(), 'features not loaded'
        return self.mapping[self.task_list[0]].shape[1]

    @property
    def n_samples(self) -> int:
        assert self.is_loaded(), 'features not loaded'
        return self.mapping[self.task_list[0]].shape[0]


class BinnedFeatureRepresentations(TaskRepresentations):
    """Torch arrays like (features x bins)"""

    def __init__(self, full_features: FullFeatureRepresentations, n_bins: int, min_q: float = 0.05,
                 max_q: float = 0.95):
        super().__init__(task_list=full_features.task_list, name=f'Binned Features (k={n_bins})')
        self.full_features = full_features
        self.n_bins = n_bins
        self.min_q = min_q
        self.max_q = max_q
        assert self.full_features.is_loaded(), 'load full features first'

    def load_representations(self):
        # get min and max per feature
        stacked = np.concatenate(tuple([self.full_features.mapping[t] for t in train_tasks]), axis=0)
        feature_mins = np.quantile(stacked, self.min_q, axis=0)
        feature_maxs = np.quantile(stacked, self.max_q, axis=0)
        # modifies to n_features x n_bins, entries are probabilities
        for task in self.task_list:
            # create bins (entries are counters)
            hist = torch.zeros((self.full_features.n_features, self.n_bins), dtype=torch.long)
            for feature_idx in range(self.full_features.n_features):
                hist[feature_idx] = torch.histc(
                    torch.tensor(self.full_features.mapping[task][:, feature_idx].flatten()).clip(
                        min=feature_mins[feature_idx], max=feature_maxs[feature_idx]),
                    bins=self.n_bins,
                    min=feature_mins[feature_idx],
                    max=feature_maxs[feature_idx])
            # normalize (make entries to probabilities)
            self.mapping[task] = hist / torch.sum(hist, dim=1, dtype=torch.double, keepdim=True)

    @property
    def n_features(self) -> int:
        return self.full_features.n_features

    def to_cuda(self):
        self.mapping = {task: t.to(torch.device('cuda')) for task, t in self.mapping.items()}

    def to_cpu(self):
        self.mapping = {task: t.to(torch.device('cpu')) for task, t in self.mapping.items()}


class AveragedFeatureRepresentations(TaskRepresentations):
    """torch arrays with size (1 x features)"""

    def __init__(self, full_features: FullFeatureRepresentations):
        super().__init__(task_list=full_features.task_list, name=f'Averaged Features')
        self.full_features = full_features
        assert self.full_features.is_loaded(), 'load full features first'

    def load_representations(self):
        for task in self.task_list:
            self.mapping[task] = torch.mean(torch.from_numpy(self.full_features.mapping[task]), dim=0).unsqueeze(0)

    @property
    def n_features(self) -> int:
        return self.full_features.n_features

    def to_cuda(self):
        self.mapping = {task: t.to(torch.device('cuda')) for task, t in self.mapping.items()}

    def to_cpu(self):
        self.mapping = {task: t.to(torch.device('cpu')) for task, t in self.mapping.items()}


class MeanAndCovarianceRepresentations(TaskRepresentations):
    """numpy arrays with size (1 x features) + (features x features)"""

    def __init__(self, full_features: FullFeatureRepresentations):
        super().__init__(task_list=full_features.task_list, name=f'Mean & Cov Features')
        self.full_features = full_features
        assert self.full_features.is_loaded(), 'load full features first'

    def load_representations(self):
        for task in self.task_list:
            mu = np.mean(self.full_features.mapping[task], axis=0)
            sigma = np.cov(self.full_features.mapping[task], rowvar=False)
            self.mapping[task] = (mu, sigma)


class FisherEmbeddingRepresentations(TaskRepresentations):
    """dicts with module name keys and torch arrays values"""

    def __init__(self, task_list: List[str] = all_tasks_including_shrunk):
        super().__init__(task_list=task_list, name='Fisher Embedding')

    def load_representations(self):
        fim_path = DATA_PATH / 'fims'
        for task in track(list(fim_path.iterdir()), transient=True):
            _task = new_to_old(task.name)
            assert _task in self.task_list, 'invalid task name'
            self.mapping[_task] = torch.load(fim_path / task.name / 'fim_0001.pkl', weights_only=True)
        assert self.is_loaded(), 'features missing'


class TagBasedRepresentations(TaskRepresentations):
    """sets of keywords plus number of samples as additional information"""

    def __init__(self, task_list: List[str] = all_tasks_including_shrunk):
        super().__init__(task_list=task_list, name='Tags')
        self.sizes = {t: 0 for t in task_list}

    def load_representations(self):
        if all([t.split(' --')[0].split('+')[0] in task_infos.num_samples for t in self.task_list]):
            # for experiments on our tasks we can use the stored infos
            for task in self.task_list:
                # undo any task splitting
                full_task = task.split(' --')[0].split('+')[0]
                self.mapping[task] = set(task_infos.keywords[full_task])
                self.sizes[task] = task_infos.num_samples[full_task]
        else:
            # fall back on actually loading local tasks
            with mml.interactive.default_file_manager() as fm:
                for task in self.task_list:
                    infos = fm.get_task_info(task.split(' --')[0].split('+')[0], preprocess='none')
                    self.mapping[task] = set(infos['keywords'])
                    self.sizes[task] = sum(infos['class_occ'].values())
