from functools import partial
from pathlib import Path

import pytest
from graphviz import ExecutableNotFound
from numpy.random import binomial
from synbn.dag import generate_dag


def test_generated_n_nodes():
    distribution = partial(binomial, n=4, p=0.5)
    dag = generate_dag(20, distribution)
    assert len(dag.vs) == 20


def test_generated_plot():
    plot_path = Path(__file__).parent / "plot.png"
    distribution = partial(binomial, n=5, p=0.5)
    dag = generate_dag(20, distribution)
    try:
        dag.plot(plot_path)
        assert plot_path.is_file()
        plot_path.unlink()
    except ExecutableNotFound:
        pass
