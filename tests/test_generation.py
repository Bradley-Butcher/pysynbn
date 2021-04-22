from functools import partial

import pandas as pd
import pytest
from baynet import DAG
from numpy.random import binomial
from synbn.generator import Generator


def _test_data() -> pd.DataFrame:
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


@pytest.mark.parametrize("assignation", ["random", "upper", "sample"])
def test_generation_from_data(assignation):
    distribution = partial(binomial, n=4, p=0.5)
    data = _test_data()
    gen = Generator.from_dataset(
        data=data,
        indegree_distribution=distribution,
        concentration=2,
        assignation=assignation,
    )
    bns = gen.sample_bns(n_bns=10)
    assert len(bns) == 10
    assert [isinstance(bn, DAG) for bn in bns]


def test_generation_marginal():
    distribution = partial(binomial, n=4, p=0.5)
    gen = Generator(
        40,
        distribution,
        [0.3, 0.2, 0.5],
        concentration=10,
        assignation="random",
    )
    bns = gen.sample_bns(10)
    assert len(bns) == 10
    assert [isinstance(bn, DAG) for bn in bns]
