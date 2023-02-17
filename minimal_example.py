import argparse

from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import (
    UtilityRandomSearch,
    PreferentialRandomSearch,
)
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1

parser = argparse.ArgumentParser(description="Run some examples!")

parser.add_argument(
    "--problem",
    choices=["zdt1", "lcbench", "tree", "forest", "xgboost"],
    default="zdt1",
)
parser.add_argument("--budget", type=int, default=50)
parser.add_argument("--optimizer", choices=["RS", "PRS", "BO"], default="BO")
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
elif args.problem == "yahpo_iaml_tree":
    print("Testing YAHPO - Instance 41146 with rpart")
    prob = YAHPO(id="iaml_rpart", instance="41146", objective_names=["auc", "ias"])
elif args.problem == "yahpo_iaml_forest":
    print("Testing YAHPO - IAML instance 41146 with with ranger")
    fix_hps = {
        "replace": "TRUE",
        "respect.unordered.factors": "ignore",
        "splitrule": "gini",
        "num_random_splits": 1,
    }
    prob = YAHPO(
        id="iaml_ranger",
        fix_hps=fix_hps,
        instance="41146",
        objective_names=["auc", "ias"],
    )
else:
    print("Testing YAHPO - Random_Bot_v2 instance 41161 with xgboost")

    fix_hps = {
        "booster": "gbtree",
        "num.impute.selected.cpo": "impute.mean",
        "repl": 1,
    }

    prob = YAHPO(
        id="rbv2_xgboost",
        fix_hps=fix_hps,
        instance="41161",
        objective_names=["auc", "memory", "timetrain"],
    )


if args.optimizer == "RS":
    print("Running Random Search")
    opt = UtilityRandomSearch(prob.get_config_space())
elif args.optimizer == "PRS":
    print("Running Preference Random Search")
    opt = PreferentialRandomSearch(prob.get_config_space())
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
