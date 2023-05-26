from typing import Dict, List, Optional, Union

import numpy as np


class DecisionMaker:
    """
    Simple Decision Maker.

    This class constructs a decision maker that has a weight for each objective.
    Preference weights need to sum to 1.

    Parameters
    ----------
    preferences: Dict, None
        Dict containing a weight for each objective
    objective_names: List, None
        List of objective names, if preferences are not given, preference values are sampled randomly
    seed: int, np.random.RandomState
        Seed used to generate the preferences if not given
    """

    def __init__(
        self,
        preferences: Optional[Dict] = None,
        objective_names: Optional[List] = None,
        seed: Optional[Union[np.random.RandomState, int]] = 42,
    ) -> None:
        assert not (preferences is None and objective_names is None), "Either preferences or objective_names must be provided."
        # FIXME: there is likely the edge case that both are given but do not match
        # if preference dict is not given, sample random preference weights and construct dict
        if preferences is None:
            np.random.seed(seed)
            pref_weights = np.random.random(len(objective_names))
            pref_weights /= sum(pref_weights)
            preferences = dict(zip(objective_names, pref_weights))
        else:
            assert sum(preferences.values()) == 1.0, "Preference weights need to sum to 1."
        self.seed = seed
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
        assert self.preferences.keys() == objectives.keys(), "Preferences and objectives need to match."

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
            Dict of objectives with associated values, that should be checked if objectives1 is preferred by DM

        Returns
        -------
        bool
            Does DM prefer objectives1 over objectives2

        """
        assert objectives1.keys() == objectives2.keys(), "Objectives need to match."

        return self._compute_utility(objectives1) > self._compute_utility(objectives2)
