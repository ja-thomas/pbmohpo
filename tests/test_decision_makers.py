import pytest

from pbmohpo.decision_makers.decision_maker import *


def test_decision_maker():
    with pytest.raises(AssertionError) as e:
        DecisionMaker()
    assert str(e.value) == "Either preferences or objective_names must be provided."

    with pytest.raises(AssertionError) as e:
        DecisionMaker(preferences={"y0": 0.5, "y1": 0.6})
    assert str(e.value) == "Preference weights need to sum to 1."

    dm = DecisionMaker(preferences={"y0": 0.5, "y1": 0.5})
    assert type(dm) == DecisionMaker
    assert type(dm.preferences) == dict

    with pytest.raises(AssertionError) as e:
        dm.compare({"y0": 0, "y1": 1}, {"y0": 1, "y2": 0})
    assert str(e.value) == "Objectives need to match."

    with pytest.raises(AssertionError) as e:
        dm.compare({"y0": 0, "y2": 1}, {"y0": 1, "y2": 0})
    assert str(e.value) == "Preferences and objectives need to match."

    assert dm.compare({"y0": 0, "y1": 1}, {"y0": 1, "y1": 0}) is False
    assert dm.compare({"y0": 1, "y1": 0}, {"y0": 0, "y1": 1}) is False
    assert dm.compare({"y0": 0.1, "y1": 1}, {"y0": 0, "y1": 1}) is True
    assert dm.compare({"y0": 0, "y1": 1}, {"y0": 0.1, "y1": 1}) is False

    # construct based on objective_names only
    dm = DecisionMaker(objective_names=["y0", "y1"])
    assert type(dm) == DecisionMaker
    assert type(dm.preferences) == dict
