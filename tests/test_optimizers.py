import pytest
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.optimizers.eubo import EUBO
from pbmohpo.benchmark import *
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.problems.zdt1 import ZDT1

@pytest.fixture(params=[RandomSearch, UtilityBayesianOptimization, EUBO])
def optimizer(request):
    return request.param

def test_optimizer(optimizer):
    problem = ZDT1()
    optimizer = optimizer(problem.get_config_space())
    decision_maker = DecisionMaker(objective_names=problem.get_objective_names(), seed=0)
    # FIXME: EUBO will only work if eval_batch_size is a multiple of 2 due to the dueling?
    # still the acquisition function is not working properly see comments in eubo.py
    # probably we want the BO fallback mechanisms to be optional so that we can actually catch and debug such errors
    benchmark = Benchmark(problem=problem, optimizer=optimizer, dm=decision_maker, eval_budget=20, dm_budget=20, eval_batch_size=2, dm_batch_size=1)
    assert type(benchmark) == Benchmark
    assert type(benchmark.archive) == Archive
    assert benchmark.archive.evaluations == []
    benchmark.run()
    # FIXME: for some reason this test fails if budget is 10 and 12 are logged
    # probably termination must be checked more properly
    assert len(benchmark.archive.evaluations) == 20

