"""We use this file as an example for some module."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd
from baynet import DAG
from baynet.parameters import ConditionalProbabilityTable
from igraph import Vertex
from scipy.stats import entropy
from tqdm import tqdm

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


def _random_permutation(amat: np.ndarray, marginals: np.ndarray) -> np.ndarray:
    return np.random.choice(range(amat.shape[0]), amat.shape[0], replace=False)


def _upper_bound_permutation(
    amat: np.ndarray, marginals: np.ndarray
) -> np.ndarray:
    perms = list(permutations(range(amat.shape[0])))
    diffs = np.zeros(len(perms))
    for i, p in enumerate(tqdm(perms)):
        diffs[i] = difficulty(marginals, amat, np.array(p))
    return perms[np.argmax(diffs)]


def _sample_permutation(amat: np.ndarray, marginals: np.ndarray) -> np.ndarray:
    return np.random.choice(range(amat.shape[0]), amat.shape[0], replace=False)


def _assign_cpt(
    dag: DAG, marginals: np.ndarray, concentration: float, assignment: str
) -> DAG:
    assignment_func = {
        "random": _random_permutation,
        "upper_bound": _upper_bound_permutation,
        "sample": _sample_permutation,
    }
    permutation = assignment_func[assignment](
        np.array(dag.get_adjacency().data), marginals
    )
    for i, vertex in zip(permutation, dag.vs):
        vertex["levels"] = list(range(len(marginals[i])))
    for i, vertex in zip(permutation, dag.vs):
        vertex["CPT"] = generate_cpt(
            vertex, marginals[i], concentration=concentration
        )
    return dag


def difficulty(marginals, amat, permutation):
    inv_e = np.array([1 / entropy(mi) for mi in marginals])[permutation]
    return np.sum(np.prod(amat * inv_e + np.ones(amat.shape), axis=1))


def generate_parameters(
    dag: DAG,
    marginals: pd.DataFrame | list[list[float]] | list[float] | np.ndarray,  # type: ignore
    concentration: float = 2,
) -> DAG:
    if isinstance(marginals, pd.DataFrame):
        marginals = _dataframe_to_marginals(marginals)
    elif isinstance(marginals, list):
        marginals = np.array(marginals)
        if marginals.shape[0] == 1:
            marginals = np.tile(marginals, len(dag.vs))
    return _assign_cpt(dag, marginals, concentration, "upper_bound")
