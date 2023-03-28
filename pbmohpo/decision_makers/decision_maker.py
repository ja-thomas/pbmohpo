from typing import Dict, List, Union

import numpy as np


class DecisionMaker:
    """
    Simple Dicision Maker.

    This class constructs a decision maker that has a weight for each objective.
    Preference weights need to sum to 1.

    Parameters
    ----------
    preferences: Dict, None
        Dict containing a weight for each objective
    objective_names: List, None
        List of objective names, if preferences are not given, preference values are sampled randomely
    """

    def __init__(
        self,
        preferences: Union[Dict, None] = None,
        objective_names: Union[List, None] = None,
    ) -> None:

        assert not (preferences is None and objective_names is None)

        # if preference dict is not given, sample random preference weights and construct dict
        if preferences is None:
            pref_weights = np.random.random(len(objective_names))
            pref_weights /= sum(pref_weights)
            preferences = dict(zip(objective_names, pref_weights))
        else:
            assert sum(preferences.values()) == 1.0

        self.preferences = preferences

    def _compute_utility(self, objectives: Dict) -> float:
        """
        Compute utility of objectives.

        Parameters
        ----------
        objectives: Dict
            Dict of objectives with associated values, needs to match preferences of DM

        Return
        ------
        float
            Utility value
        """
        assert self.preferences.keys() == objectives.keys()

        pref_vals = [
            self.preferences[k] * objectives[k] for k in self.preferences.keys()
        ]
        return sum(pref_vals)

    def compare(self, objectives1: Dict, objectives2: Dict) -> bool:
        """
        Check if the DM prefers objectives 1 over objectives 2.

        Parameters
        ----------
        objectives1: Dict
            Dict of objectives with associated values that should be checked for preference
        objectives2: Dict
            Dict of objectives with associated values, that should be checked if objectives1 is prefered by DM

        Returns
        -------
        bool
            Does DM prefer objectives1 over objectives2

        """
        assert objectives1.keys() == objectives2.keys()

        return self._compute_utility(objectives1) > self._compute_utility(objectives2)
