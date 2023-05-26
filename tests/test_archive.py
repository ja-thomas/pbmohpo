import numpy as np
import torch
from ConfigSpace import Configuration

from pbmohpo.benchmark import *
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.problems.zdt1 import ZDT1


def test_archive():
    archive = Archive()
    assert type(archive) == Archive


def test_archive_after_run():
    problem = ZDT1(dimension=2)
    optimizer = RandomSearch(problem.get_config_space())
    decision_maker = DecisionMaker(
        objective_names=problem.get_objective_names(), seed=0
    )
    benchmark = Benchmark(
        problem=problem,
        optimizer=optimizer,
        dm=decision_maker,
        eval_budget=10,
        dm_budget=10,
        eval_batch_size=1,
        dm_batch_size=1,
    )
    assert type(benchmark.archive) == Archive
    assert benchmark.archive.evaluations == []
    benchmark.run()
    assert len(benchmark.archive.evaluations) == 10
    assert type(benchmark.archive.evaluations[0]) == Evaluation
    assert type(benchmark.archive.evaluations[0].config) == Configuration
    assert type(benchmark.archive.evaluations[0].objectives) == dict
    assert len(benchmark.archive.evaluations[0].objectives) == 2
    assert benchmark.archive.evaluations[0].objectives["y0"] is not None
    assert benchmark.archive.evaluations[0].objectives["y1"] is not None
    assert benchmark.archive.evaluations[0].utility is not None
    np_values = benchmark.archive.to_numpy()
    assert type(np_values) == tuple
    assert len(np_values) == 2
    assert type(np_values[0]) == np.ndarray
    assert type(np_values[1]) == np.ndarray
    assert np_values[0].shape == (10, 2)
    assert np_values[1].shape == (10,)
    torch_values = benchmark.archive.to_torch()
    assert type(torch_values) == tuple
    assert len(torch_values) == 2
    assert type(torch_values[0]) == torch.Tensor
    assert type(torch_values[1]) == torch.Tensor
    assert torch_values[0].shape == (10, 2)
    assert torch_values[1].shape == (
        10,
        1,
    )  # FIXME: not consistent with np_values[1] shape
    assert (
        type(benchmark.archive.max_utility) == np.float64
    )  # FIXME: could also check other types e.g. objectives, np_values, torch_values
    assert type(benchmark.archive.incumbents) == list
