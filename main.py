import argparse
import logging
import os
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow import log_artifact, log_metric, log_metrics, log_params

from config import get_cfg_defaults
from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO, qEUBO
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.lgboml import LgbOpenML
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.utils import visualize_archives


def run_pbmohpo_bench(
    config,
    visualize: bool = False,
    use_mlflow: bool = False,
    save_archive: bool = False,
):
    """
    Run a preferential bayesian hyperparameter optimization benchmark as
    specified in the config file.

    Parameters
    ----------
    config: str
    path to config.yaml file. File should be in the following format.
    --
    PROBLEM:
      PROBLEM_TYPE: "yahpo"
      ID: "iaml_ranger"
      INSTANCE: "1067"
      OBJECTIVE_NAMES: ["auc", "nf"]
      OBJECTIVE_SCALING_FACTORS: [1, 21]

    FIXED_HPS:
      TRAINSIZE: ("trainsize", 1)
      REPLACE: ("replace", "TRUE")
      RESPECT_UNORDERED_FACTORS: ("respect.unordered.factors", "ignore")
      SPLITRULE: ("splitrule", "gini")
    --
    For options than can be set in config files, please see config.py.

    visualize: bool
    Specify whether to create plots (with default configuration) for
    benchmark.

    use_mlflow: bool
    Should the experiment be tracked with mlflow.

    save_archive: bool
    Should the archive be saved to a file.
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
        logging.info(f"objective scaling: {config.PROBLEM.OBJECTIVE_SCALING_FACTORS}")

        fixed_hyperparams = {}
        for hyperparam in config.FIXED_HPS:
            if config.FIXED_HPS[hyperparam][0]:
                hp_name = config.FIXED_HPS[hyperparam][0]
                hp_value = config.FIXED_HPS[hyperparam][1]
                fixed_hyperparams[hp_name] = hp_value

        scaling = dict(
            zip(
                config.PROBLEM.OBJECTIVE_NAMES, config.PROBLEM.OBJECTIVE_SCALING_FACTORS
            )
        )

        prob = YAHPO(
            id=config.PROBLEM.ID,
            fix_hps=fixed_hyperparams,
            instance=str(config.PROBLEM.INSTANCE),
            objective_names=config.PROBLEM.OBJECTIVE_NAMES,
            objective_scaling_factors=scaling,
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
    elif config.OPTIMIZER.OPTIMIZER_TYPE == "qEUBO":
        logging.info("Running Bayesian Optimization on Pairwise Comparisons (qEUBO)")
        opt = qEUBO(prob.get_config_space())
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

    # FIXME: maybe also save archive to mlflow
    if save_archive:
        logging.info("Saving archive")
        x, y = archive.to_numpy()
        df = pd.DataFrame(y, columns=["utility"])
        df["best"] = df["utility"].cummax()
        if config.PROBLEM.PROBLEM_TYPE == "yahpo":
            df["prob"] = (
                config.PROBLEM.PROBLEM_TYPE
                + "_"
                + config.PROBLEM.ID
                + "_"
                + str(config.PROBLEM.INSTANCE)
            )
        else:
            df["prob"] = config.PROBLEM.PROBLEM_TYPE
        df["opt"] = config.OPTIMIZER.OPTIMIZER_TYPE
        df["seed"] = str(config.DECISION_MAKER.SEED)
        df["iter"] = range(1, len(df) + 1)
        if len(df) is not config.BUDGET.EVAL_BUDGET:
            logging.warning("Archive has not been fully populated")
        path = "experiment_results"
        subpath = df["prob"][0]
        path = os.path.join(path, subpath)
        if not os.path.exists(path):
            os.makedirs(path)
        file = (
            os.path.join(path, df["prob"][0] + "_" + df["opt"][0])
            + "_"
            + str(df["seed"][0])
            + "_"
            + datetime.now().strftime("%y%m%d%H%M%S")
            + ".csv"
        )
        df.to_csv(file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify experiment to run")

    parser.add_argument(
        "-c",
        "--config",
        help="Config file of experiment",
        default="./experiment_configs/templates/iaml_xgboost_1067.yaml",
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

    parser.add_argument(
        "-s",
        "--save_archive",
        help="Should the archive be saved to a file",
        action="store_true",
        dest="save_archive",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.loglevel)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()
    logging.debug(cfg)

    run_pbmohpo_bench(
        cfg,
        visualize=args.visualize,
        use_mlflow=args.use_mlflow,
        save_archive=args.save_archive,
    )
