from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO, qEUBO
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.utils import visualize_archives

prob = YAHPO(
    "iaml_rpart",
    instance="1067",
    objective_names=["auc", "nf"],
    fix_hps={"trainsize": 1},
    objective_scaling_factors={"auc": 1, "nf": 21},
    seed=0,
)

dm = DecisionMaker(objective_names=prob.get_objective_names(), seed=0)

print("Decision Maker Preference Scores:")
print(dm.preferences)

opt = UtilityBayesianOptimization(prob.get_config_space())
bench = Benchmark(
    prob, opt, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
print("Running Utility BO")
bench.run()

opt2 = RandomSearch(prob.get_config_space())
bench2 = Benchmark(
    prob, opt2, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
print("Running RS")
bench2.run()

opt3 = EUBO(prob.get_config_space())
bench3 = Benchmark(
    prob, opt3, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
print("Running EUBO")
bench3.run()

opt4 = qEUBO(prob.get_config_space())
bench4 = Benchmark(
    prob, opt4, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
print("Running qEUBO")
bench4.run()

print("Best Configuration:")
print(bench.archive.evaluations[bench.archive.incumbents[0]])
print(bench2.archive.evaluations[bench2.archive.incumbents[0]])
print(bench3.archive.evaluations[bench3.archive.incumbents[0]])
print(bench4.archive.evaluations[bench4.archive.incumbents[0]])

visualize_archives(
    [bench.archive, bench2.archive, bench3.archive, bench4.archive],
    legend_elements=["BO", "RS", "EUBO", "qEUBO"],
)
