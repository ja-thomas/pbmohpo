import argparse

from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1

parser = argparse.ArgumentParser(description="Run some examples!")

parser.add_argument("--problem", choices=["zdt1", "yahpo"], default="zdt1")
parser.add_argument("--budget", type=int, default=50)
parser.add_argument("--optimizer", choices=["RS", "BO"], default="BO")
parser.add_argument("--dimension", type=int, default=10)

args = parser.parse_args()

budget = args.budget
optimizer = args.optimizer
problem = args.problem

if args.problem == "zdt1":
    print("Testing ZDT1")
    prob = ZDT1(dimension=args.dimension)
else:
    print("Testing YAHPO")
    prob = YAHPO(
        id="lcbench", instance="3945", objective_names=["time", "val_accuracy"]
    )

if args.optimizer == "RS":
    print("Running Random Search")
    opt = RandomSearch(prob.get_config_space())
else:
    print("Running Bayesian Optimization on Utility Scores")
    opt = UtilityBayesianOptimization(prob.get_config_space())

dm = DecisionMaker(objective_names=prob.get_objective_names())

print("Decision Maker Preference Scores:")
print(dm.preferences)

bench = Benchmark(prob, opt, dm, budget)
bench.run()

print(f"Best Configuration found in iteration [{bench.archive.incumbents[0]}]:")
print(bench.archive.data[bench.archive.incumbents[0]])
