import argparse
import logging
import tempfile

import matplotlib.pyplot as plt
import mlflow
from mlflow import log_artifact, log_metric, log_metrics, log_params

from config import get_cfg_defaults
from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.lgboml import LgbOpenML
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.utils import visualize_archives


def run_pbmohpo_bench(config, visualize: bool = False, use_mlflow: bool = False):
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

    visualize: bool
    Specify whether to create plots (with default configuration) for
    benchmark.

    use_mlflow: bool
    Should the experiment be tracked with mlflow.
    """

    if use_mlflow:
        mlflow.set_experiment(config.NAME.EXPERIMENT_NAME)
        for c_item in config.values():
            log_params(c_item)

    if config.PROBLEM.PROBLEM_TYPE == "zdt1":
        logging.info("Testing ZDT1")
        prob = ZDT1(dimension=config.PROBLEM.DIMENSIONS)

    elif config.PROBLEM.PROBLEM_TYPE == "yahpo":
        logging.info("Testing YAHPO")
        logging.info(f"id: {config.PROBLEM.ID}")
        logging.info(f"instance: {config.PROBLEM.INSTANCE}")
        logging.info(f"objectives: {config.PROBLEM.OBJECTIVE_NAMES}")

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

    elif config.PROBLEM.PROBLEM_TYPE == "lgboml":
        logging.info(
            f"Testing LightGBM Tuning on OpenML Task {config.PROBLEM.OML_TASK}"
        )
        prob = LgbOpenML(task_id=config.PROBLEM.OML_TASK)

    if config.OPTIMIZER.OPTIMIZER_TYPE == "RS":
        logging.info("Running Random Search")
        opt = RandomSearch(prob.get_config_space())
    elif config.OPTIMIZER.OPTIMIZER_TYPE == "BO":
        logging.info("Running Bayesian Optimization on Utility Scores")
        opt = UtilityBayesianOptimization(prob.get_config_space())
    else:
        logging.info("Running Bayesian Optimization on Pairwise Comparisons")
        opt = EUBO(prob.get_config_space())

    dm = DecisionMaker(
        objective_names=prob.get_objective_names(), seed=config.DECISION_MAKER.SEED
    )

    logging.info("Decision Maker Preference Scores:")
    logging.info(dm.preferences)

    bench = Benchmark(
        prob,
        opt,
        dm,
        eval_budget=config.BUDGET.EVAL_BUDGET,
        dm_budget=config.BUDGET.DM_BUDGET,
        eval_batch_size=config.BATCH_SIZE.EVAL_BATCH_SIZE,
        dm_batch_size=config.BATCH_SIZE.DM_BATCH_SIZE,
    )
    bench.run()

    archive = bench.archive
    best_eval = archive.evaluations[archive.incumbents[0]]

    if use_mlflow:
        log_metric("utility", best_eval.utility)
        log_metrics(best_eval.objectives)

    logging.info(f"Best Configuration found in iteration [{archive.incumbents[0]}]:")
    logging.info(best_eval)

    if visualize:
        visualize_archives(archive_list=[archive])
        if mlflow:
            logging.info("Storing trace")
            plt.savefig(f"{tempfile.gettempdir()}/trace.png")
            log_artifact(f"{tempfile.gettempdir()}/trace.png")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify experiment to run")

    parser.add_argument(
        "-c",
        "--config",
        help="Config file of experiment",
        default="./experiment_configs/iaml_ranger.yaml",
        dest="config",
    )

    parser.add_argument(
        "-d",
        "--debug",
        help="Print debugging statements",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_const",
        dest="loglevel",
        const=logging.INFO,
    )

    parser.add_argument(
        "-p",
        "--plot",
        help="creates plots (default config) of conducted benchmark",
        action="store_true",
        dest="visualize",
    )

    parser.add_argument(
        "-m",
        "--mlflow",
        help="Should experiment be tracked by mlflow",
        action="store_true",
        dest="use_mlflow",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    logging.debug(cfg)

    run_pbmohpo_bench(cfg, visualize=args.visualize, use_mlflow=args.use_mlflow)
