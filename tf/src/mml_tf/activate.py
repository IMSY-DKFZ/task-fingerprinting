from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin


# register plugin configs
class MMLTaskFingerprintingSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Sets the search path for mml with copied config files
        search_path.append(
            provider="mml-tf", path="pkg://mml_tf.configs"
        )


Plugins.instance().register(MMLTaskFingerprintingSearchPathPlugin)
