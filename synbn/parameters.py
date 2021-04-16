"""We use this file as an example for some module."""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd
from baynet import DAG
from baynet.parameters import ConditionalProbabilityTable
from igraph import Vertex

__all__ = ["generate_parameters", "generate_cpt"]


def generate_cpt(
    vertex: Vertex, marginal_distribution: np.ndarray, concentration: float
) -> ConditionalProbabilityTable:
    cpt = ConditionalProbabilityTable(vertex=vertex)
    alphas = np.array(marginal_distribution) * concentration
    cpt.sample_parameters(alpha=alphas)
    return cpt


def _dataframe_to_marginals(data: pd.DataFrame) -> np.ndarray:
    proportions = [
        list(data.groupby([col])[col].value_counts().values / len(data))
        for col in data.columns
    ]
    return np.array(proportions, dtype="object")


def generate_parameters(
    dag: DAG,
    marginals: pd.DataFrame | list[list[float]] | list[float] | np.ndarray,
    concentration: float = 2,
) -> DAG:
    if isinstance(marginals, pd.DataFrame):
        marginals = _dataframe_to_marginals(marginals)
    elif isinstance(marginals, list):
        marginals = np.array(marginals)
        if marginals.shape[0] == 1:
            marginals = np.tile(marginals, len(dag.vs))
    for i, vertex in enumerate(dag.vs):
        vertex["levels"] = list(range(len(marginals[i])))
    for i, vertex in enumerate(dag.vs):
        vertex["CPT"] = generate_cpt(
            vertex, marginals[i], concentration=concentration
        )
    return dag
