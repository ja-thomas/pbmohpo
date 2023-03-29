import argparse

from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1

parser = argparse.ArgumentParser(description="Run some examples!")

parser.add_argument(
    "--problem",
    choices=[
        "zdt1",
        "yahpo_lcbench",
        "yahpo_iaml_rpart",
        "yahpo_iaml_ranger",
        "yahpo_rbv2_xgboost",
    ],
    default="zdt1",
)
parser.add_argument("--eval_budget", type=int, default=50)
parser.add_argument("--dm_budget", type=int, default=50)
parser.add_argument("--optimizer", choices=["RS", "BO"], default="BO")
parser.add_argument("--dimension", type=int, default=10)

args = parser.parse_args()

eval_budget = args.eval_budget
dm_budget = args.dm_budget
optimizer = args.optimizer
problem = args.problem

if args.problem == "zdt1":
    print("Testing ZDT1")
    prob = ZDT1(dimension=args.dimension)
elif args.problem == "yahpo_lcbench":
    print("Testing YAHPO - lcbench instance 3945")
    fix_hps = {"epoch": 52}
    prob = YAHPO(
        id="lcbench",
        fix_hps=fix_hps,
        instance="3945",
        objective_names=["time", "val_accuracy"],
    )
elif args.problem == "yahpo_iaml_rpart":
    print("Testing YAHPO - IAML instance 41146 with rpart")
    fix_hps = {"trainsize": 1}
    prob = YAHPO(
        id="iaml_rpart",
        fix_hps=fix_hps,
        instance="41146",
        objective_names=["auc", "ias"],
    )
elif args.problem == "yahpo_iaml_ranger":
    print("Testing YAHPO - IAML instance 41146 with with ranger")
    fix_hps = {
        "trainsize": 1,
        "replace": "TRUE",
        "respect.unordered.factors": "ignore",
        "splitrule": "gini",
    }
    prob = YAHPO(
        id="iaml_ranger",
        fix_hps=fix_hps,
        instance="41146",
        objective_names=["auc", "nf", "ias"],
    )
elif args.problem == "yahpo_rbv2_xgboost":
    print("Testing YAHPO - Random_Bot_v2 instance 41161 with xgboost")

    fix_hps = {
        "trainsize": 1,
        "booster": "gbtree",
        "num.impute.selected.cpo": "impute.mean",
        "repl": 10,
    }

    prob = YAHPO(
        id="rbv2_xgboost",
        fix_hps=fix_hps,
        instance="41161",
        objective_names=["auc", "timetrain"],
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

bench = Benchmark(prob, opt, dm, eval_budget=eval_budget, dm_budget=dm_budget)
bench.run()

archive = bench.archive

print(f"Best Configuration found in iteration [{archive.incumbents[0]}]:")
print(archive.data[archive.incumbents[0]])
