from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from mml.core.data_loading.file_manager import MMLFileManager
from mml.core.data_loading.augmentations.albumentations import AlbumentationsAugmentationModule
from mml.core.scripts.exceptions import TaskNotFoundError
from mml.core.scripts.schedulers.base_scheduler import AFTER_SCHEDULER_INIT_HOOKS, AbstractBaseScheduler
from mml.core.scripts.schedulers.train_scheduler import TrainingScheduler
from omegaconf import OmegaConf


# register plugin configs
class MMLTaskFingerprintingSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(
            provider="mml-tf", path="pkg://mml_tf.configs"
        )


Plugins.instance().register(MMLTaskFingerprintingSearchPathPlugin)

# the loading of auto augmentation created policies requires a register
MMLFileManager.add_assignment_path(
    obj_cls=dict,
    key="aa",
    path=Path("PROJ_PATH") / "AA" / "TASK_NAME" / "policy.json",
    enable_numbering=True,
    reusable=True,
)

# we also need to introduce backward compatibility in the loading of such a policy through replacing the default
# augmentation parser of the AlbumentationsAugmentationModule
backup_cfg_parser = AlbumentationsAugmentationModule.from_cfg

def new_from_cfg(aug_config):
    import albumentations as A
    if len(aug_config) > 0 and aug_config[-1].name == "LoadAA":
        return backup_cfg_parser(OmegaConf.create(aug_config[:-1])) + [A.load(aug_config[-1].path)]
    else:
        return backup_cfg_parser(aug_config)

AlbumentationsAugmentationModule.from_cfg = staticmethod(new_from_cfg)

class AALoadingTrainScheduler(TrainingScheduler):
    """Small modification of default scheduler that after loading injects the correct AA path into config."""

    def insert_aa_path_into_cfg(self) -> None:
        """Inject the path of the AA policy.json into the config."""
        if 'pipeline' in self.cfg.augmentations.cpu:
            for aug_cfg in self.cfg.augmentations.cpu.pipeline:
                if aug_cfg.name == "LoadAA":
                    try:
                        struct = self.get_struct(aug_cfg.source)
                    except TaskNotFoundError:
                        raise RuntimeError(f'Could not load AA from a task that is not present in experiment. Please '
                                           f'add {aug_cfg.source} to task_list.')
                    if "aa" not in struct.paths:
                        raise RuntimeError(f'No AA loaded for task {aug_cfg.source}. Make sure to set '
                                           f'+reuse.aa=SOME_PROJECT that has an existing AA policy for this task.')
                    aug_cfg.path = struct.paths['aa']

    def after_preparation_hook(self):
        """Is called after experiment setup."""
        super().after_preparation_hook()
        self.insert_aa_path_into_cfg()



