import copy

import pytest
from ConfigSpace import ConfigurationSpace
from yahpo_gym import benchmark_set

from pbmohpo.problems.yahpo import *
from pbmohpo.problems.zdt1 import *


def test_zdt1():
    with pytest.raises(AssertionError) as e:
        ZDT1(dimension=1, seed=0)
    assert str(e.value) == "Dimension must be >= 2."
    problem = ZDT1(dimension=2, seed=0)
    assert type(problem) == ZDT1
    configspace = problem.get_config_space()
    assert type(configspace) == ConfigurationSpace
    assert configspace.get_hyperparameter("x0").lower == 0.0
    assert configspace.get_hyperparameter("x0").upper == 1.0
    assert configspace.get_hyperparameter("x1").lower == 0.0
    assert configspace.get_hyperparameter("x1").upper == 1.0
    assert problem.n_objectives == 2
    assert problem.get_objective_names() == ["y0", "y1"]
    ys = problem(configspace.get_default_configuration())
    assert type(ys) == dict
    assert len(ys) == 2
    assert ys["y0"] == 0.0
    assert ys["y1"] == -1.0


def test_yahpo():
    problem = YAHPO(
        "iaml_rpart",
        instance="41146",
        objective_names=["auc", "nf"],
        fix_hps={"trainsize": 1},
        seed=0,
    )
    assert type(problem) == YAHPO
    configspace = problem.get_config_space()
    assert type(configspace) == ConfigurationSpace
    assert configspace.get_hyperparameter_names() == [
        "cp",
        "maxdepth",
        "minbucket",
        "minsplit",
    ]
    assert problem.n_objectives == 2
    assert problem.get_objective_names() == ["auc", "nf"]

    config = configspace.get_default_configuration()
    config_old = copy.deepcopy(config)
    ys = problem(config)
    assert config_old.get_dictionary() == config.get_dictionary()
    assert type(ys) == dict
    assert len(ys) == 2
    assert (ys["auc"] >= 0) & (ys["auc"] <= 1)
    assert ys["nf"] < 0.0
    # same result with yahpo_gym
    bench = benchmark_set.BenchmarkSet("iaml_rpart")
    bench.set_instance("41146")
    config = configspace.get_default_configuration().get_dictionary()
    config.update({"task_id": bench.instance, "trainsize": 1})
    ys_yahpo = bench.objective_function(config)
    assert ys_yahpo[0]["auc"] == ys["auc"]
    assert ys_yahpo[0]["nf"] == -ys["nf"]

    # test without fix_hps
    problem = YAHPO(
        "iaml_rpart", instance="41146", objective_names=["auc", "nf"], seed=0
    )
    assert type(problem) == YAHPO
    configspace = problem.get_config_space()
    assert type(configspace) == ConfigurationSpace
    assert configspace.get_hyperparameter_names() == [
        "cp",
        "maxdepth",
        "minbucket",
        "minsplit",
        "trainsize",
    ]
    assert problem.n_objectives == 2
    assert problem.get_objective_names() == ["auc", "nf"]
    ys = problem(configspace.get_default_configuration())
    assert type(ys) == dict
    assert len(ys) == 2
    assert (ys["auc"] >= 0) & (ys["auc"] <= 1)
    assert ys["nf"] < 0.0
