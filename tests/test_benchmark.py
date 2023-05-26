import pytest

from pbmohpo.benchmark import *
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1


@pytest.fixture(params=[ZDT1, YAHPO])
def problem(request):
    return request.param


def test_benchmark(problem, **kwargs):
    if problem == YAHPO:
        required = {
            "id": "iaml_rpart",
            "instance": "41146",
            "objective_names": ["auc", "nf"],
        }
        kwargs.update(required)
    problem = problem(**kwargs)
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
    assert type(benchmark) == Benchmark
    assert type(benchmark.archive) == Archive
    assert benchmark.archive.evaluations == []
    benchmark.run()
    assert len(benchmark.archive.evaluations) == 10
