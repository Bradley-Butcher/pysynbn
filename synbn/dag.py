"""We use this file as an example for some module."""
from __future__ import annotations

from typing import Callable, cast

import numpy as np
from baynet import DAG

__all__ = ["generate_dag"]


def _shuffle(positive: int, total: int, padding: int) -> np.ndarray:
    positive = np.min([positive, total])
    result = np.concatenate([np.ones(positive), np.zeros(total - positive)])
    np.random.shuffle(result)  # Uses standard uniform shuffling
    return cast(np.ndarray, np.concatenate([np.zeros(padding), result]))


def generate_dag(
    nodes: int, distribution: Callable[[int], list[int]], seed: int = 1  # type: ignore
) -> DAG:
    """Generate a DAG

    Args:
        nodes (int): The number of nodes in the DAG
        distribution(Callable): Any discrete sampling method
        seed (int): The random seed for the stochastic processes in the function

    Returns:
        DAG: A generated DAG

    Examples:
        .. code:: python
            >>> from functools import partial
            >>> from numpy.random import binomial
            >>> dist = partial(binomial, n=5, p=0.5)
            >>> bn = generate_dag(20, dist)
            >>> len(bn.vs)
            20
    """
    np.random.seed(seed)
    in_degree_samples = sorted(distribution(size=nodes), reverse=True)  # type: ignore
    adj_mat = np.zeros((nodes, nodes))
    for i in range(nodes):
        n_parents = in_degree_samples[i]
        adj_mat[:, i] = _shuffle(n_parents, nodes - (i + 1), i + 1)
    return DAG.from_amat(adj_mat.T, [str(i) for i in range(nodes)])
