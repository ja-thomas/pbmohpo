from typing import Dict, List, Optional, Union

import ConfigSpace as CS
import numpy as np

from pbmohpo.problems.problem import Problem


class ZDT1(Problem):
    """
    Synthetic Test Function ZDT1

    Can be scaled to an arbitrary number of dimensions and has two objectives.

    Since we are always maximizing we're taking the negative function values.

    Parameters
    ----------
    seed: int, np.random.RandomState
        Seed passed to the problem
    dimension: int
        Number of dimensions, needs to be at least 2.
    """

    objective_names = ["y0", "y1"]

    def __init__(
        self, seed: Optional[Union[np.random.RandomState, int]] = 42, dimension: int = 2
    ) -> None:
        super().__init__(seed)
        assert dimension >= 2, "Dimension must be >= 2."
        self.dimension = dimension

    def get_config_space(self) -> CS.ConfigurationSpace:
        cs = CS.ConfigurationSpace(seed=self.seed)
        for i in range(self.dimension):
            cs.add_hyperparameter(
                CS.UniformFloatHyperparameter(
                    "x" + str(i), lower=0, upper=1, default_value=0, log=False
                )
            )
        return cs

    def get_objective_names(self) -> List:
        return self.objective_names

    def __call__(
        self,
        x: CS.Configuration,
        seed: Optional[Union[np.random.RandomState, int]] = None,
    ) -> Dict:
        f1 = x["x0"]  # objective 1
        g = 1 + 9 * np.sum(x.get_array()[1:] / (self.dimension - 1))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h  # objective 2

        return {self.objective_names[0]: -f1, self.objective_names[1]: -f2}
