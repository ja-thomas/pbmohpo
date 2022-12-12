from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.benchmark import Benchmark

dim = 10
budget = 100

prob = ZDT1(dimension=dim)
dm = DecisionMaker(objective_names=prob.get_objective_names())
rs = UtilityBayesianOptimization(prob.get_config_space())

bench = Benchmark(prob, rs, dm, budget)
bench.run()
