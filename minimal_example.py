import argparse

import matplotlib.pyplot as plt

from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import \
    UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.utils import visualize_archives

parser = argparse.ArgumentParser(description="Run some examples!")

parser.add_argument(
    "--problem", choices=["zdt1", "yahpo_lcbench", "yahpo_rbv2"], default="zdt1"
)
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
elif args.problem == "yahpo_lcbench":
    print("Testing YAHPO - lcbench instance 3945")
    prob = YAHPO(
        id="lcbench", instance="3945", objective_names=["time", "val_accuracy"]
    )
else:
    print("Testing YAHPO - Random Bot v2 instance")
    prob = YAHPO(id="iaml_rpart", instance="41146", objective_names=["auc", "ias"])


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

# fig = visualize_archives([bench.archive], ["incumbent", "utilities"])
# plt.show()

print(f"Best Configuration found in iteration [{bench.archive.incumbents[0]}]:")
print(bench.archive.data[bench.archive.incumbents[0]])
