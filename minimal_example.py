from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1

dim = 10
budget = 50

print("Testing ZDT1")

prob = ZDT1(dimension=dim)
dm = DecisionMaker(objective_names=prob.get_objective_names())
opt = UtilityBayesianOptimization(prob.get_config_space())

bench = Benchmark(prob, opt, dm, budget)
bench.run()


print("Testing YAHPO")

targets = ["time", "val_accuracy"]

prob = YAHPO(id="lcbench", instance="3945", objective_names=targets)
dm = DecisionMaker(objective_names=prob.get_objective_names())
opt = UtilityBayesianOptimization(prob.get_config_space())

bench = Benchmark(prob, opt, dm, budget)
bench.run()
