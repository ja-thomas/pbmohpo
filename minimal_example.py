from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO
from pbmohpo.problems.zdt1 import ZDT1

prob = ZDT1(dimension=2)
opt = EUBO(prob.get_config_space(), duels_per_eval=5)
dm = DecisionMaker(objective_names=prob.get_objective_names())

print("Decision Maker Preference Scores:")
print(dm.preferences)

bench = Benchmark(
    prob, opt, dm, eval_budget=20, dm_budget=200, eval_batch_size=2, dm_batch_size=1
)
bench.run()
