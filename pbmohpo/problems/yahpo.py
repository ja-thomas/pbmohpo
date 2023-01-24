from typing import Dict, List, Union

import ConfigSpace as CS
import numpy as np
from yahpo_gym import benchmark_set

from pbmohpo.problems.problem import Problem


class YAHPO(Problem):
    def __init__(
        self,
        id: str,
        instance: str,
        objective_names: List,
        seed: Union[int, np.random.RandomState, None] = 42,
    ) -> None:
        super().__init__(seed)
        self.benchmark = benchmark_set.BenchmarkSet(id)
        self.benchmark.set_instance(instance)
        self.objective_names = objective_names

    def get_config_space(self) -> CS.ConfigurationSpace:
        return self.benchmark.get_opt_space()

    def get_objective_names(self) -> List:
        return self.objective_names

    def __call__(
        self, x: CS.Configuration, seed: Union[np.random.RandomState, int, None] = None
    ) -> Dict:
        val_dict = self.benchmark.objective_function(x)[0]

        positions = [
            self.benchmark.config.config.get("y_names").index(obj)
            for obj in self.get_objective_names()
        ]

        factor = [
            self.benchmark.config.config.get("y_minimize")[pos] for pos in positions
        ]
        factor = [-1 if fac else 1 for fac in factor]
        val_dict = {
            key: factor[el] * val_dict[key]
            for el, key in enumerate(self.get_objective_names())
        }
        return val_dict


if __name__ == "__main__":

    targets = ["time", "val_accuracy"]

    prob = YAHPO(id="lcbench", instance="3945", objective_names=targets)

    conf = prob.get_config_space().sample_configuration()

    prob(conf)
