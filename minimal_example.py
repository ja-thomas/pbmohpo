from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO, qEUBO
from pbmohpo.optimizers.utility_bayesian_optimization import UtilityBayesianOptimization
from pbmohpo.optimizers.random_search import RandomSearch
from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.problems.yahpo import YAHPO
from pbmohpo.utils import visualize_archives

# prob = ZDT1(dimension=2)
prob = YAHPO(
    "iaml_rpart",
    instance="1067",
    objective_names=["auc", "nf"],
    fix_hps={"trainsize": 1},
    objective_scaling_factors={"auc": 1, "nf": 21},
    seed=0,
)

# opt = EUBO(prob.get_config_space())
# opt = qEUBO(prob.get_config_space())
opt = UtilityBayesianOptimization(prob.get_config_space())

dm = DecisionMaker(objective_names=prob.get_objective_names(), seed=0)

print("Decision Maker Preference Scores:")
print(dm.preferences)

bench = Benchmark(
    prob, opt, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
bench.run()

opt2 = RandomSearch(prob.get_config_space())

bench2 = Benchmark(
    prob, opt2, dm, eval_budget=100, dm_budget=100, eval_batch_size=2, dm_batch_size=1
)
bench2.run()

print("Best Configuration:")
print(bench.archive.evaluations[bench.archive.incumbents[0]])
print("Best Configuration:")
print(bench2.archive.evaluations[bench2.archive.incumbents[0]])

visualize_archives(
    [bench.archive, bench2.archive], legend_elements=["BO", "Random Search"]
)
