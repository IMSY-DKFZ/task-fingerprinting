from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent.parent  # this projects base path

FIG_PATH = BASE_PATH / 'figures'
DATA_PATH = BASE_PATH / 'data'
CACHE_PATH = BASE_PATH / 'cache'

MML_PROJECTS_VALIDATION = {
    # Baseline performances
    'full_baseline': 'pami2_raw_03',  # mml 0.8.1
    'shrunk_baseline': 'pami2_raw_shrunk_10',  # mml 0.8.1
    # EXP 1
    'arch_search': 'pami2_t_arch_search_02',  # mml 0.8.1
    'arch_shrunk': 'pami2_t_arch_infer_02',  # mml 0.8.1
    # EXP 2
    'transfer': 'pami2_t_transfer_20',  # mml 0.8.1
    # EXP 3
    'aa_infer': 'pami2_t_aa_infer_02',  # mml 0.8.1
    # EXP 4
    'multi_task': 'test_multi_balanced_test_split_10',  # needs at least mml 0.9.0 for fix #9
    'multi_shrunk': 'test_multi_balanced_shrunk_test_split_10',  # needs at least mml 0.9.0 for fix #9
}

MML_PROJECTS_TRAIN = {
    # Baseline performances
    'full_baseline': 'pami2_raw_03',  # mml 0.8.1
    'shrunk_baseline': 'pami2_raw_shrunk_10',  # mml 0.8.1
    # EXP 1
    'arch_search': 'pami2_arch_search_02',  # mml 0.8.1
    'arch_shrunk': 'pami2_arch_infer_02',  # mml 0.8.1
    # EXP 2
    'transfer': 'pami2_transfer_20',  # mml 0.8.1
    # EXP 3
    'aa_infer': 'pami2_aa_infer_02',  # mml 0.8.1
    # EXP 4
    'multi_task': 'test_multi_balanced_10',  # needs at least mml 0.9.0 for fix #9
    'multi_shrunk': 'test_multi_balanced_shrunk_10',  # needs at least mml 0.9.0 for fix #9
}
