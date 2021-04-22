from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from baynet import DAG

__all__ = ["Generator"]

from synbn.dag import generate_dag
from synbn.parameters import (
    PermutationType,
    Permutator,
    dataframe_to_marginals,
    generate_parameters,
)


class Generator:
    def __init__(
        self,
        nodes: int,
        indegree_distribution: Callable[[int], list[int]],
        marginals: list[list[float]] | list[float] | np.ndarray,  # type: ignore
        concentration: float = 2,
        assignation: str | PermutationType = "random",
        seed: int = 1,
    ):
        self.nodes = nodes
        self.indegree_distribution = indegree_distribution
        self.marginals = marginals
        self.concentration = concentration
        self.assignation = (
            PermutationType(assignation)
            if isinstance(assignation, str)
            else assignation
        )
        self.seed = seed

    @classmethod
    def from_dataset(
        cls,
        data: pd.DataFrame,
        indegree_distribution: Callable[[int], list[int]],
        concentration: float = 2,
        assignation: str | PermutationType | Permutator = "random",
        seed: int = 1,
    ) -> Generator:
        nodes = len(data.columns)
        marginals = dataframe_to_marginals(data)
        generator = Generator(
            nodes=nodes,
            indegree_distribution=indegree_distribution,
            marginals=marginals,
            concentration=concentration,
            assignation=assignation,
            seed=seed,
        )
        return generator

    @classmethod
    def from_bn(
        cls,
        dag: DAG,
        concentration: float = 2,
        assignation: str | PermutationType | Permutator = "random",
        seed: int = 1,
    ) -> Generator:
        pass

    def sample_dag(self) -> DAG:
        dag = generate_dag(self.nodes, self.indegree_distribution, self.seed)
        self.seed += 1
        return dag

    def sample_dags(self, n_dags: int = 10) -> list[DAG]:
        return [self.sample_dag() for i in range(n_dags)]

    def sample_bn(self, with_permutators: bool = False) -> DAG:
        dag = self.sample_dag()
        bn, perm = generate_parameters(
            dag, self.marginals, self.concentration, self.assignation
        )
        self.seed += 1
        if with_permutators:
            return bn, perm
        else:
            return bn

    def sample_bns(
        self, n_bns: int = 10, with_permutators: bool = False
    ) -> list[DAG]:
        return [self.sample_bn(with_permutators) for i in range(n_bns)]
