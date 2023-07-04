import copy
from typing import Dict, List, Optional, Union

import ConfigSpace as CS
import numpy as np
from yahpo_gym import benchmark_set

from pbmohpo.problems.problem import Problem
from pbmohpo.utils import remove_hp_from_cs


class YAHPO(Problem):
    """
    YAHPO Gym Problem.

    This class wraps YAHPO Gym (https://github.com/slds-lmu/yahpo_gym/).

    Parameters
    ----------
    id: str
        Benchmark class from YAHPO
    instance: str
        Instance of benchmark
    objective_names: List[str]
        Objectives to optimize
    fix_hps: Dict
        Dictionary of fixed HPs that should not be optimized
    seed: int, np.random.RandomState
        Seed passed to the problem
    """

    def __init__(
        self,
        id: str,
        instance: str,
        objective_names: List,
        fix_hps: dict = None,
        seed: Optional[Union[np.random.RandomState, int]] = 42,
    ) -> None:
        super().__init__(seed)
        if fix_hps is None:
            fix_hps = {}
        self.fix_hps = fix_hps
        self.benchmark = benchmark_set.BenchmarkSet(id)
        self.benchmark.set_instance(instance)
        self.objective_names = objective_names

    def get_config_space(self) -> CS.ConfigurationSpace:
        # Remove instance information from config space
        csn = copy.deepcopy(self.benchmark.get_opt_space())
        hps_to_remove = self.fix_hps
        hps_to_remove.update(
            {self.benchmark.config.instance_names: self.benchmark.instance}
        )

        cs = remove_hp_from_cs(csn, remove_hp_dict=hps_to_remove)
        return cs

    def get_objective_names(self) -> List:
        return self.objective_names

    def __call__(
        self,
        x: CS.Configuration,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Dict:
        # Add instance information to configuration
        x = copy.deepcopy(x.get_dictionary())
        x.update({self.benchmark.config.instance_names: self.benchmark.instance})
        # Add fixed HPs back to configuration
        x.update(self.fix_hps)

        val_dict = self.benchmark.objective_function(x)[0]

        # check which objectives are minimized and invert the objective value
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
