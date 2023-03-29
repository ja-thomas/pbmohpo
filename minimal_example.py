import argparse

from pbmohpo.benchmark import Benchmark
from pbmohpo.decision_makers.decision_maker import DecisionMaker
from pbmohpo.optimizers.eubo import EUBO
from pbmohpo.problems.zdt1 import ZDT1

prob = ZDT1(dimension=2)
opt = EUBO(prob.get_config_space())
dm = DecisionMaker(objective_names=prob.get_objective_names())

print("Decision Maker Preference Scores:")
print(dm.preferences)

bench = Benchmark(prob, opt, dm, eval_budget=5, dm_budget=15)
bench.run()
