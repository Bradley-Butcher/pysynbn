from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from baynet import DAG
from numpy.random import binomial
from synbn.dag import generate_dag
from synbn.parameters import _dataframe_to_marginals, generate_parameters


def _test_dag(nodes: int) -> DAG:
    distribution = partial(binomial, n=4, p=0.5)
    dag = generate_dag(nodes, distribution)
    return dag


def _test_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0] * 10 + [1] * 30,
            "B": [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10,
        }
    )


def _test_data_large(nodes: int) -> pd.DataFrame:
    data = np.random.choice(3, (5, nodes))
    return pd.DataFrame(data, columns=[str(xi) for xi in range(data.shape[1])])


def test_dataframe_to_marginals():
    marginals = _dataframe_to_marginals(_test_data())
    true = np.array([[0.25, 0.75], [0.25, 0.25, 0.25, 0.25]], dtype="object")
    np.testing.assert_equal(marginals, true)


def test_bn_has_parameters():
    dag = _test_dag(8)
    data = _test_data_large(8)
    bn = generate_parameters(dag, data)
    assert [v["CPT"] for v in bn.vs]
