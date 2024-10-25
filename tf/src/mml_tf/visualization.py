"""
General plotting utils for the task fingerprinting project.
"""

from itertools import cycle
from typing import Tuple, Sequence, Dict, Any, List, Optional

import numpy as np
import plotly.graph_objects as go
from PIL import ImageColor

from mml_tf.distances import map_dist2printable

_EXPS = []
_DISTANCE_MEASURES = []

_palette = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#ffe119', '#fabebe', '#f032e6',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#ffd8b1', '#000075', '#808080',
            '#ffffff', '#bcf60c', '#000000']
ALL_COLORS = [ImageColor.getcolor(color, "RGB") for color in _palette]
GREEN = ImageColor.getcolor('#3cb44b', 'RGB')
GREEN_VAR_1 = ImageColor.getcolor('#3cb487', "RGB")
GREEN_VAR_2 = ImageColor.getcolor('#69b43c', "RGB")


def init_colors(exp: List[str], distance_measures: List[str]) -> None:
    """Init exp and distance measures for color stability."""
    global _EXPS, _DISTANCE_MEASURES
    _EXPS = exp
    _DISTANCE_MEASURES = distance_measures


def _distribute_colors(list_or_enum: Sequence, pivots: Sequence) -> Dict[Any, Tuple[int, int, int]]:
    """
    Generates a map of elements to colors, based on a sequence and up to three pivot elements.

    :param list_or_enum: list or enum class
    :param pivots: pivot elements, will be shown green
    """
    non_pivots = [elem for elem in list_or_enum if elem not in pivots]
    color_cycle = cycle(ALL_COLORS)
    _map = {elem: next(color_cycle) for idx, elem in enumerate(non_pivots)}
    if pivots is not None:
        for p in pivots:
            _map[p] = GREEN
    return _map


def stringify(rgb_tuple: Tuple[int, int, int], opacity: Optional[float] = None) -> str:
    """
    Turns an RGB tuple into plotly readable color string with optional opacity.

    :param rgb_tuple: tuple of three ints representing RGB values
    :param opacity: optional opacity as float
    :return: color code as either "rgba(red, green, blue, alpha)" if opacity is given else "rgb(red, green, blue)"
    """
    if opacity is not None:
        return "rgba" + str(rgb_tuple)[:-1] + f", {opacity})"
    else:
        return "rgb" + str(rgb_tuple)


def get_exp_color(exp: str, opacity: Optional[float] = None) -> str:
    """
    Gives consistent colors to identical exps.

    :param exp: experiment name
    :param opacity: if provided this will we used as alpha value for the opacity
    :return: color code as either "rgba(red, green, blue, alpha)" if opacity is given else "rgb(red, green, blue)"
    """
    return stringify(rgb_tuple=_distribute_colors(list_or_enum=_EXPS, pivots=[])[exp], opacity=opacity)


def get_dist_measure_color(distance_measure: str, opacity: Optional[float] = None) -> str:
    """
    Gives consistent colors to identical distance_measure methods.

    :param distance_measure: distance measure
    :param opacity: if provided this will we used as alpha value for the opacity
    :return: color code as either "rgba(red, green, blue, alpha)" if opacity is given else "rgb(red, green, blue)"
    """
    return stringify(rgb_tuple=_distribute_colors(
        list_or_enum=_DISTANCE_MEASURES,
        pivots=[map_dist2printable[d] for d in
                ['KLD-PP:NS-W:TS-100-BINS', 'KLD-PP:NS-W:SN-1000-BINS', 'KLD-PP:NS-1000-BINS']])[distance_measure],
                     opacity=opacity)


def add_aggregate_results(infos: Dict[str, List[List[float]]],
                          budget: int = 10,
                          ci: bool = True,
                          opacity: float = 0.15,
                          line_width: float = 4.,
                          row=None, col=None, fig=None):
    """Useful for line plots with option for shaded uncertainty areas."""
    x = list(np.arange(1, budget + 1, 1))
    x_rev = x[::-1]

    def make_lines(results, line_name, color):
        y = list(np.average(results, axis=0))
        if ci:
            y_upper = list(np.average(results, axis=0) + np.std(results, axis=0))
            y_lower = list(np.average(results, axis=0) - np.std(results, axis=0))
            y_lower = y_lower[::-1]
            fig.add_trace(go.Scatter(
                x=x + x_rev,
                y=y_upper + y_lower,
                fill='toself',
                showlegend=False,
                name=line_name,
                line_color='rgba(255,255,255,0)',
                fillcolor="rgba" + str(color)[:-1] + f", {opacity})"

            ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            name=line_name,
            line_color="rgb" + str(color),
            line_width=line_width,
            showlegend=(row == col == 1),
            legendrank=
            {'BCED': 1200, 'CED': 1150, 'ssFED': 1100, 'FED': 1050, 'RANDOM': 900, 'SEMANTIC': 910, 'KLD': 920,
             'EMD': 930, 'VDNA': 940, 'P2L': 950, 'FID': 960}[name]

        ), row=row, col=col)

    # pallete = [ImageColor.getcolor(color, "RGB") for color in px.colors.qualitative.Plotly]
    for line_idx, name in enumerate(infos):
        color = get_dist_measure_color(name)
        results = np.stack((infos[name]))
        make_lines(results, name, color)
    return fig
