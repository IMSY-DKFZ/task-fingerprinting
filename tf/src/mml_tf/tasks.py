"""Holds information on the tasks used by the project"""
from typing import List
from mml.core.scripts.exceptions import TaskNotFoundError
import mml.interactive
from mml_tf.paths import CACHE_PATH
from sqlalchemy.testing.plugin.plugin_base import warnings

train_tasks = (
    'lapgyn4_anatomical_structures', 'lapgyn4_surgical_actions', 'lapgyn4_instrument_count',
    'lapgyn4_anatomical_actions',
    'sklin2_skin_lesions', 'identify_nbi_infframes', 'laryngeal_tissues', 'nerthus_bowel_cleansing_quality',
    'stanford_dogs_image_categorization', 'svhn', 'caltech101_object_classification',
    'caltech256_object_classification',
    'cifar10_object_classification', 'cifar100_object_classification', 'mnist_digit_classification',
    'emnist_digit_classification', 'hyperkvasir_anatomical-landmarks', 'hyperkvasir_pathological-findings',
    'hyperkvasir_quality-of-mucosal-views', 'hyperkvasir_therapeutic-interventions', 'cholec80_grasper_presence',
    'cholec80_bipolar_presence', 'cholec80_hook_presence', 'cholec80_scissors_presence', 'cholec80_clipper_presence',
    'cholec80_irrigator_presence', 'cholec80_specimenbag_presence', 'derm7pt_skin_lesions')

test_tasks = (
    'idle_action_recognition', 'barretts_esophagus_diagnosis', 'brain_tumor_classification',
    'mednode_melanoma_classification', 'brain_tumor_type_classification',
    'chexpert_enlarged_cardiomediastinum', 'chexpert_cardiomegaly', 'chexpert_lung_opacity',
    'chexpert_lung_lesion', 'chexpert_edema', 'chexpert_consolidation', 'chexpert_pneumonia',
    'chexpert_atelectasis', 'chexpert_pneumothorax', 'chexpert_pleural_effusion', 'chexpert_pleural_other',
    'chexpert_fracture', 'chexpert_support_devices',
    'pneumonia_classification', 'ph2-melanocytic-lesions-classification',
    'covid_xray_classification', 'isic20_melanoma_classification', 'deep_drid_dr_level',
    'shenzen_chest_xray_tuberculosis', 'crawled_covid_ct_classification', 'deep_drid_quality',
    'deep_drid_clarity', 'deep_drid_field', 'deep_drid_artifact', 'kvasir_capsule_anatomy',
    'kvasir_capsule_content', 'kvasir_capsule_pathologies',
    'breast_cancer_classification_v2',
    'eye_condition_classification', 'mura_xr_wrist', 'mura_xr_shoulder', 'mura_xr_humerus', 'mura_xr_hand',
    'mura_xr_forearm', 'mura_xr_finger', 'mura_xr_elbow',
    'bean_plant_disease_classification',
    'aptos19_blindness_detection')

all_tasks = train_tasks + test_tasks

source_tasks = all_tasks
target_tasks = test_tasks

if (CACHE_PATH / 'task_infos.csv').exists():
    task_infos = mml.interactive.planning.AllTasksInfos.from_csv(CACHE_PATH / 'task_infos.csv')
else:
    try:
        task_infos = mml.interactive.get_task_infos(task_list=list(all_tasks), dims='pami2_base_02')
        task_infos.store_csv(CACHE_PATH / 'task_infos.csv')
    except TaskNotFoundError as e:
        warnings.warn(f'No task_infos.csv found in cache and neither seem all tasks installed properly. Will try to '
                      f'fall back without task_infos available. Some functionality of mml_tf will not be acessible.')
        attrs = [
            "task_types",
            "num_classes",
            "num_samples",
            "domains",
            "imbalance_ratios",
            "datasets",
            "keywords",
            "dimensions",
            "max_resolution",
            "min_resolution",
        ]
        kwargs = {attr: {} for attr in attrs}
        task_infos = AllTasksInfos(small_tasks=[], medium_tasks=[], large_tasks=[], **kwargs)


# tasks sharing images should not be inspected together
task_groups = {
    'chexpert': [
        'chexpert_cardiomegaly',
        'chexpert_lung_opacity',
        'chexpert_lung_lesion',
        'chexpert_edema',
        'chexpert_consolidation',
        'chexpert_pneumonia',
        'chexpert_atelectasis',
        'chexpert_pneumothorax',
        'chexpert_pleural_effusion',
        'chexpert_pleural_other',
        'chexpert_fracture',
        'chexpert_support_devices',
        'chexpert_enlarged_cardiomediastinum'
    ],
    'cholec80': [
        'cholec80_grasper_presence',
        'cholec80_bipolar_presence',
        'cholec80_hook_presence',
        'cholec80_scissors_presence',
        'cholec80_clipper_presence',
        'cholec80_irrigator_presence',
        'cholec80_specimenbag_presence',
        'lapgyn4_instrument_count'
    ],
    'deep_drid': [
        'deep_drid_dr_level',
        'deep_drid_quality',
        'deep_drid_clarity',
        'deep_drid_field',
        'deep_drid_artifact'
    ],
    'xray': [
        'pneumonia_classification',
        'covid_xray_classification'
    ]
}


# returns valid source tasks respecting phase and task_groups
def get_valid_sources(target_task: str, min_samples: int = 1) -> List[str]:
    valid_sources = train_tasks if target_task in train_tasks else all_tasks
    # check if target is in a group
    group_id = None
    for group_name, group_list in task_groups.items():
        if target_task in group_list:
            group_id = group_name
            break
    # if so further reduce valid sources
    if group_id:
        valid_sources = [t for t in valid_sources if t not in task_groups[group_id]]
    return [t for t in valid_sources if t != target_task and task_infos.num_samples[t] >= min_samples]


# the task shrinking operation only makes sense if there are certain restrictions met (< 40 classes and > 1000 samples)
non_shrinkable_tasks = ['sklin2_skin_lesions',
                        'identify_nbi_infframes',
                        'stanford_dogs_image_categorization',
                        'caltech101_object_classification',
                        'caltech256_object_classification',
                        'cifar100_object_classification',
                        'emnist_digit_classification',
                        'derm7pt_skin_lesions',
                        'barretts_esophagus_diagnosis',
                        'mednode_melanoma_classification',
                        'ph2-melanocytic-lesions-classification',
                        'shenzen_chest_xray_tuberculosis',
                        'crawled_covid_ct_classification',
                        'breast_cancer_classification_v2',
                        'eye_condition_classification']

shrinkable_tasks = [t for t in all_tasks if t not in non_shrinkable_tasks]

# we keep using the old mml tagging style (' --TAG') consistently throughout the project
all_tasks_including_shrunk = list(all_tasks) + [t + ' --shrink_train 800' for t in all_tasks if t in shrinkable_tasks]

shrink_map = {}  # will hold a map from base task to shrunk version
for t in all_tasks:
    if t in non_shrinkable_tasks:
        shrink_map[t] = t
    else:
        shrink_map[t] = t + ' --shrink_train 800'


def new_to_old(task: str) -> str:
    """Turns a tagged task in new style format into old style."""
    if '+' in task:
        return task.replace('+shrink_train?800', ' --shrink_train 800')
    return task


def old_to_new(task: str) -> str:
    """Turns a tagged task in old style format into new style."""
    if ' --' in task:
        return task.replace(' --shrink_train 800', '+shrink_train?800')
    return task


"""Turns a task from the internal identifier to the paper representation (e.g. T17)"""
paper_id_map = {
    # train tasks
    'lapgyn4_anatomical_structures': 'T06',
    'lapgyn4_surgical_actions': 'T07',
    'lapgyn4_instrument_count': 'T08',
    'lapgyn4_anatomical_actions': 'T09',
    'sklin2_skin_lesions': 'T23',
    'identify_nbi_infframes': 'T27',
    'laryngeal_tissues': 'T28',
    'nerthus_bowel_cleansing_quality': 'T01',
    'stanford_dogs_image_categorization': 'T17',
    'svhn': 'T18',
    'caltech101_object_classification': 'T19',
    'caltech256_object_classification': 'T20',
    'cifar10_object_classification': 'T21',
    'cifar100_object_classification': 'T22',
    'mnist_digit_classification': 'T25',
    'emnist_digit_classification': 'T26',
    'hyperkvasir_anatomical-landmarks': 'T02',
    'hyperkvasir_pathological-findings': 'T03',
    'hyperkvasir_quality-of-mucosal-views': 'T04',
    'hyperkvasir_therapeutic-interventions': 'T05',
    'cholec80_grasper_presence': 'T10',
    'cholec80_bipolar_presence': 'T11',
    'cholec80_hook_presence': 'T12',
    'cholec80_scissors_presence': 'T13',
    'cholec80_clipper_presence': 'T14',
    'cholec80_irrigator_presence': 'T15',
    'cholec80_specimenbag_presence': 'T16',
    'derm7pt_skin_lesions': 'T24',
    # test tasks
    'idle_action_recognition': 'T53',
    'barretts_esophagus_diagnosis': 'T54',
    'brain_tumor_classification': 'T57',
    'mednode_melanoma_classification': 'T60',
    'brain_tumor_type_classification': 'T58',
    'chexpert_enlarged_cardiomediastinum': 'T38',
    'chexpert_cardiomegaly': 'T39',
    'chexpert_lung_opacity': 'T40',
    'chexpert_lung_lesion': 'T41',
    'chexpert_edema': 'T37',
    'chexpert_consolidation': 'T29',
    'chexpert_pneumonia': 'T30',
    'chexpert_atelectasis': 'T31',
    'chexpert_pneumothorax': 'T32',
    'chexpert_pleural_effusion': 'T33',
    'chexpert_pleural_other': 'T34',
    'chexpert_fracture': 'T35',
    'chexpert_support_devices': 'T36',
    'pneumonia_classification': 'T42',
    'ph2-melanocytic-lesions-classification': 'T61',
    'covid_xray_classification': 'T44',
    'isic20_melanoma_classification': 'T62',
    'deep_drid_dr_level': 'T63',
    'shenzen_chest_xray_tuberculosis': 'T43',
    'crawled_covid_ct_classification': 'T59',
    'deep_drid_quality': 'T64',
    'deep_drid_clarity': 'T65',
    'deep_drid_field': 'T66',
    'deep_drid_artifact': 'T67',
    'kvasir_capsule_anatomy': 'T69',
    'kvasir_capsule_content': 'T70',
    'kvasir_capsule_pathologies': 'T71',
    'breast_cancer_classification_v2': 'T55',
    'eye_condition_classification': 'T56',
    'mura_xr_wrist': 'T45',
    'mura_xr_shoulder': 'T46',
    'mura_xr_humerus': 'T47',
    'mura_xr_hand': 'T48',
    'mura_xr_forearm': 'T49',
    'mura_xr_finger': 'T50',
    'mura_xr_elbow': 'T51',
    'bean_plant_disease_classification': 'T52',
    'aptos19_blindness_detection': 'T68'
}
