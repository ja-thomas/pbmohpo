import pytest
from ConfigSpace import ConfigurationSpace
from yahpo_gym import benchmark_set

from pbmohpo.utils import remove_hp_from_cs


def test_remove_hp_from_cs():
    bench = benchmark_set.BenchmarkSet("fair_xgboost")
    old_cs = bench.get_opt_space()
    remove_hp_dict = {
        "task_id": "31",
        "trainsize": 1,
        "pre_post": "none",
        "booster": "gbtree",
    }
    new_cs = remove_hp_from_cs(old_cs, remove_hp_dict=remove_hp_dict)
    assert type(new_cs) == ConfigurationSpace
    assert set(new_cs.get_hyperparameter_names()) == set(
        [
            "alpha",
            "colsample_bytree",
            "colsample_bylevel",
            "eta",
            "gamma",
            "lambda",
            "max_depth",
            "min_child_weight",
            "nrounds",
            "subsample",
        ]
    )
