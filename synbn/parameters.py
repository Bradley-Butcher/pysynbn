"""We use this file as an example for some module."""
from __future__ import annotations

import warnings
from enum import Enum
from itertools import permutations

import numpy as np
import pandas as pd
import seaborn as sns
from baynet import DAG
from baynet.parameters import ConditionalProbabilityTable
from igraph import Vertex
from matplotlib import pyplot as plt
from scipy.stats import entropy, gaussian_kde, skellam
from sympy import Eq, solve
from sympy.abc import x, y

__all__ = [
    "generate_parameters",
    "generate_cpt",
    "Permutator",
    "PermutationType",
    "dataframe_to_marginals",
    "binomial_mv",
]


class PermutationType(Enum):
    RANDOM = "random"
    UPPER_BOUND = "upper"
    SAMPLE = "sample"


class Permutator:
    def __init__(
        self, permutation_type: PermutationType, marginals, adj_matrix, **kwargs
    ):
        self.permutation_type = permutation_type
        self.marginals = marginals
        self.adj_matrix = adj_matrix
        assignment_func = {
            "random": self._random_permutation,
            "upper": self._upper_bound_permutation,
            "sample": self._sample_permutation,
        }
        self.permutation = assignment_func[permutation_type.value]()

    @classmethod
    def from_dag(
        cls,
        dag: DAG,
        marginals: np.ndarray,
        permutation_type: PermutationType | str,
    ):
        permutation_type = (
            PermutationType(permutation_type)
            if isinstance(permutation_type, str)
            else permutation_type
        )
        return Permutator(
            permutation_type, marginals, np.array(dag.get_adjacency().data)
        )

    def difficulty(self, permutation):
        inv_e = np.array([1 / entropy(list(mi)) for mi in self.marginals])[
            permutation
        ]
        return np.sum(
            np.prod(
                self.adj_matrix * inv_e + np.ones(self.adj_matrix.shape), axis=1
            )
        )

    def _random_permutation(self) -> np.ndarray:
        return np.random.choice(
            range(self.adj_matrix.shape[0]),
            self.adj_matrix.shape[0],
            replace=False,
        )

    def _upper_bound_permutation(self) -> np.ndarray:
        out_degree_order = np.argsort(np.sum(self.adj_matrix, axis=0))[::-1]
        inv_e = np.argsort(
            np.array([1 / entropy(list(mi)) for mi in self.marginals])
        )[::-1]
        perm = np.zeros(self.adj_matrix.shape[0], dtype=np.int32)
        for i, e in enumerate(inv_e):
            perm[e] = out_degree_order[i]
        return perm

    def _build_difficulty_distribution(self, samples: int) -> np.ndarray:
        permutation_dict = {}
        if np.math.factorial(self.adj_matrix.shape[0]) < samples:
            # DO FULL PERMUTATION
            pass
        for i in range(samples):
            permutation = np.random.permutation(self.adj_matrix.shape[0])
            permutation_dict[self.difficulty(permutation)] = permutation
        self.permutation_dict = permutation_dict

    def _sample_permutation(self, samples: int = 100):
        self._build_difficulty_distribution(samples=samples)
        dist = np.array(list(self.permutation_dict.keys()))
        if len(dist) == 1:
            return self.permutation_dict[dist[0]]
        kde = gaussian_kde(dist)
        sample = kde.resample(1)[0][0]
        idx = np.argmin(np.abs(dist - sample))
        return self.permutation_dict[dist[idx]]

    def difficulty_distribution(self):
        assert (
            self.permutation_dict
        ), "Permutation Dict not found. Use the sampling method."
        sns.distplot(list(self.permutation_dict.keys()))
        plt.show()


def generate_cpt(
    vertex: Vertex, marginal_distribution: np.ndarray, concentration: float
) -> ConditionalProbabilityTable:
    cpt = ConditionalProbabilityTable(vertex=vertex)
    alphas = np.array(marginal_distribution, dtype=np.float64) * concentration
    cpt.sample_parameters(alpha=alphas)
    return cpt


def dataframe_to_marginals(data: pd.DataFrame) -> np.ndarray:
    proportions = [
        list(data.groupby([col])[col].value_counts().values / len(data))
        for col in data.columns
    ]
    return np.array(proportions, dtype="object")


def _assign_cpt(
    dag: DAG, marginals: np.ndarray, concentration: float, assignment_type: str
) -> DAG & Permutator:
    permutator = Permutator.from_dag(dag, marginals, assignment_type)
    for i, vertex in zip(permutator.permutation, dag.vs):
        vertex["levels"] = list(range(len(marginals[i])))
    for i, vertex in zip(permutator.permutation, dag.vs):
        vertex["CPD"] = generate_cpt(
            vertex, marginals[i], concentration=concentration
        )
    return dag, permutator


def generate_parameters(
    dag: DAG,
    marginals: pd.DataFrame | list[list[float]] | list[float] | np.ndarray,  # type: ignore
    concentration: float = 2,
    assignation: str | PermutationType | Permutator = "random",
    seed: int = 1,
) -> DAG & Permutator:
    np.random.seed(seed)
    if isinstance(marginals, pd.DataFrame):
        marginals = dataframe_to_marginals(marginals)
    elif isinstance(marginals, list):
        marginals = np.array(marginals)
        if marginals.ndim == 1:
            warnings.warn(
                "Assignation rendered irrelevant due to one dimensional marginal."
            )
            marginals = np.tile(marginals, (len(dag.vs), 1))
    return _assign_cpt(dag, marginals, concentration, assignation)


def binomial_mv(mean: float, variance: float, size: int):
    sol = solve([Eq(x * y, mean), Eq(x * y * (1 - y), variance)])[0]
    res = {s: sol[s].evalf() for s in sol}
    return np.random.binomial(n=res[x], p=res[y], size=size)
