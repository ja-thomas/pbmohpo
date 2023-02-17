from config import get_cfg_defaults
from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import \
    UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.problems.zdt1 import ZDT1


def run_pbmohpo_bench(config):

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
                hp_name = config.FIXED_HPS[hyperparam][1]
                hp_value = config.FIXED_HPS[hyperparam][2]
                fixed_hyperparams[hp_name] = hp_value

        prob = YAHPO(
            id=config.PROBLEM.ID,
            fix_hps=fixed_hyperparams,
            instance=str(config.PROBLEM.INSTANCE),
            objective_names=config.PROBLEM.OBJECTIVE_NAMES,
        )

    if config.OPTIMIZER.OPTIMIZER_TYPE == "RS":
        print("Running Random Search")
        opt = RandomSearch(prob.get_config_space())
    else:
        print("Running Bayesian Optimization on Utility Scores")
        opt = UtilityBayesianOptimization(prob.get_config_space())

    dm = DecisionMaker(objective_names=prob.get_objective_names())

    print("Decision Maker Preference Scores:")
    print(dm.preferences)

    bench = Benchmark(prob, opt, dm, config.BUDGET.BUDGET_AMOUNT)
    bench.run()

    print(f"Best Configuration found in iteration [{bench.archive.incumbents[0]}]:")
    print(bench.archive.data[bench.archive.incumbents[0]])


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    # cfg.merge_from_file("minimal_example.yaml")
    cfg.merge_from_file("./experiment_configs/iaml_ranger.yaml")
    cfg.freeze()
    print(cfg)

    run_pbmohpo_bench(cfg)
