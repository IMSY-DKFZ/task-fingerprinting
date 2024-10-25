"""Sets up order, renaming and coloring for imaging domains."""

from dataclasses import dataclass
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt
import mml.interactive
import mml_similarity.visualization.plot_2D as plot_2D


@dataclass
class DomainColorInfo:
    legend_map: Dict[int, str]  # maps entry number to "pretty" domain
    color_map: Dict[str, int]  # maps task to entry number
    color_discrete_map: Dict[int, str]  # maps "pretty" domain to hex color str


def init_domain_visualization(all_tasks: List[str]) -> DomainColorInfo:
    """Sets up necessary visualization for imaging domains of tasks."""
    color_map, legend_map = plot_2D.create_color_mapping(task_list=mml.interactive.get_task_structs(all_tasks),
                                                         criteria='domain',
                                                         task_clusters=None)
    REPLACEMENTS = {'Cataract_surgery': 'ophthalmic microscopy', 'Chars_or_digits': 'handwritings',
                    'Confocal laser endomicroscopy': 'confocal laser endomicroscopy',
                    'Ct_scan': 'CT', 'Dermatoscopy': 'dermatoscopy', 'Fundus_photography': 'fundus photography',
                    'Gastroscopy_colonoscopy': 'gastro & colonoscopy', 'Laparoscopy': 'laparoscopy',
                    'Laryngoscopy': 'laryngoscopy',
                    'Mri_scan': 'MRI', 'Natural_objects': 'natural images', 'Other': 'other', 'X_ray': 'X-ray',
                    'Capsule endoscopy': 'capsule endoscopy', 'Ultrasound': 'ultrasound'}
    for k in legend_map:
        if legend_map[k] in REPLACEMENTS:
            legend_map[k] = REPLACEMENTS[legend_map[k]]
    # re-arrangement of legend elements
    final_order = [
        # endoscopy
        'confocal laser endomicroscopy', 'gastro & colonoscopy', 'laparoscopy', 'laryngoscopy', 'capsule endoscopy',
        # radiology
        'CT', 'MRI', 'X-ray', 'ultrasound',
        # microscopy
        'ophthalmic microscopy', 'dermatoscopy', 'fundus photography',
        # natural images
        'natural images', 'handwritings', 'other'
    ]
    for k in color_map:
        color_map[k] = final_order.index(legend_map[color_map[k]])
    legend_map = {idx: val for idx, val in enumerate(final_order)}
    all_colors = list(set(color_map.values()))
    # Color mapping
    palette = plt.get_cmap('jet')
    normalizer = matplotlib.colors.Normalize(vmin=0, vmax=max(all_colors))
    scalar_map = matplotlib.cm.ScalarMappable(norm=normalizer, cmap=palette)
    color_discrete_map = {legend_map[color_map[t]]: matplotlib.colors.to_hex(scalar_map.to_rgba(color_map[t])) for t in
                          color_map}

    for k in legend_map:
        if legend_map[k] in REPLACEMENTS:
            legend_map[k] = REPLACEMENTS[legend_map[k]]

    return DomainColorInfo(legend_map=legend_map, color_map=color_map, color_discrete_map=color_discrete_map)
