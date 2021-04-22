from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from baynet import DAG
from matplotlib import pyplot as plt
from numpy.random import binomial
from synbn.dag import generate_dag
from synbn.parameters import (
    binomial_mv,
    dataframe_to_marginals,
    generate_parameters,
)


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


def _test_data_manual_large() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [0] * 7 + [1] * 3,
            "B": [0] * 2 + [1] * 8,
            "C": [0] * 1 + [1] * 9,
            "D": [0] * 4 + [1] * 6,
            "E": [0] * 5 + [1] * 5,
            "F": [0] * 4 + [1] * 6,
            "G": [0] * 8 + [1] * 2,
        }
    )


def _test_data_large(nodes: int) -> pd.DataFrame:
    data = np.random.choice(3, (10, nodes))
    return pd.DataFrame(data, columns=[str(xi) for xi in range(data.shape[1])])


def test_dataframe_to_marginal():
    marginals = dataframe_to_marginals(_test_data())
    true = np.array([[0.25, 0.75], [0.25, 0.25, 0.25, 0.25]], dtype="object")
    np.testing.assert_equal(marginals, true)


@pytest.mark.parametrize("assignation", ["random", "upper", "sample"])
def test_bn_has_parameters(assignation: str):
    dag = _test_dag(7)
    data = _test_data_manual_large()
    bn, perms = generate_parameters(dag, data, assignation=assignation)
    assert [v["CPT"] for v in bn.vs]


def test_bionmial_reparam():
    mean, var = 3, 1
    samples = binomial_mv(mean, var, 1000)
    assert np.round(np.mean(samples)) == mean
