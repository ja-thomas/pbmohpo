import argparse

from config import get_cfg_defaults
from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.optimizer import PreferenceOptimizer
from pbmohpo.optimizers.random_search import UtilityRandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import \
    UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1


def run_pbmohpo_bench(config):
    """
    Run a preferential bayesian hyperparameter optimization benchmark as
    specified in the config file.

    Parameters
    ----------
    config: str
    path to config.yaml file. File should be in the following format.
    --
    PROBLEM:
      DIMENSIONS: 7
      PROBLEM_TYPE: "yahpo"

    FIXED_HPS:
      REPLACE: (True, "replace", "TRUE")
      SPLITRULE: (True, "splitulre", "gini")
    --
    For options than can be set in config files, please see config.py
    """
    if config.PROBLEM.PROBLEM_TYPE == "zdt1":
        print("Testing ZDT1")
        prob = ZDT1(dimension=config.PROBLEM.DIMENSIONS)

    elif config.PROBLEM.PROBLEM_TYPE == "yahpo":

        print("Testing YAHPO")
        print(f"id: {config.PROBLEM.ID}")
        print(f"instance: {config.PROBLEM.INSTANCE}")
        print(f"objectives: {config.PROBLEM.OBJECTIVE_NAMES}")

        fixed_hyperparams = {}
        for hyperparam in config.FIXED_HPS:
            if config.FIXED_HPS[hyperparam][0]:
                hp_name = config.FIXED_HPS[hyperparam][0]
                hp_value = config.FIXED_HPS[hyperparam][1]
                fixed_hyperparams[hp_name] = hp_value

        prob = YAHPO(
            id=config.PROBLEM.ID,
            fix_hps=fixed_hyperparams,
            instance=str(config.PROBLEM.INSTANCE),
            objective_names=config.PROBLEM.OBJECTIVE_NAMES,
        )

    if config.OPTIMIZER.OPTIMIZER_TYPE == "RS":
        print("Running Random Search")
        opt = UtilityRandomSearch(prob.get_config_space())
    else:
        print("Running Bayesian Optimization on Utility Scores")
        opt = UtilityBayesianOptimization(prob.get_config_space())

    dm = DecisionMaker(objective_names=prob.get_objective_names())

    print("Decision Maker Preference Scores:")
    print(dm.preferences)

    bench = Benchmark(prob, opt, dm, config.BUDGET.BUDGET_AMOUNT)
    bench.run()

    archive = (
        bench.archive.to_utility_archive()
        if issubclass(type(opt), PreferenceOptimizer)
        else bench.archive
    )

    print(f"Best Configuration found in iteration [{archive.incumbents[0]}]:")
    print(archive.data[archive.incumbents[0]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Specify the experiment to run")

    parser.add_argument(
        "-p",
        default="./experiment_configs/iaml_ranger.yaml",
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.p)
    cfg.freeze()
    print(cfg)

    run_pbmohpo_bench(cfg)
