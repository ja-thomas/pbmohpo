from pbmohpo.problems.zdt1 import ZDT1
from pbmohpo.decision_makers.decision_maker import Decision_maker
from pbmohpo.optimizers.random_search import Random_search
from pbmohpo.benchmark import Benchmark

dim = 10
budget = 100

prob = ZDT1(dimension=dim)
dm = Decision_maker(objective_names=prob.get_objective_names())
rs = Random_search(prob.get_config_space())

bench = Benchmark(prob, rs, dm, budget)
bench.run()
