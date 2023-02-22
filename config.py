from pbmohpo.problems.zdt1 import ZDT1

from yacs.config import CfgNode as CN

_C = CN()

_C.PROBLEM = CN()
_C.PROBLEM.PROBLEM_TYPE = 'zdt1'

# ZDT1 Settings
_C.PROBLEM.DIMENSIONS = 10

# YAHPO Setting
_C.PROBLEM.ID = "lcbench"
_C.PROBLEM.INSTANCE = 3945
_C.PROBLEM.OBJECTIVE_NAMES = ["time", "val_accuracy"]

# Fixed HPs for YAHPO
_C.FIXED_HPS = CN()

_C.FIXED_HPS.REPLACE = (None, None)
_C.FIXED_HPS.SPLITRULE = (None, None)
_C.FIXED_HPS.RESPECT_UNORDERED_FACTORS = (None, None)
_C.FIXED_HPS.NUM_RANDOM_SPLITS = (None, None)

_C.FIXED_HPS.BOOSTER = (None, None)
_C.FIXED_HPS.NUM_IMPUTE_SELECTED_CPOR = (None, None)
_C.FIXED_HPS.REPL = (None, None)

_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER_TYPE = 'BO'

_C.DECISION_MAKER = CN()
_C.DECISION_MAKER.DECISION_MAKER_TYPE = "DecisionMaker"

_C.BUDGET = CN()
_C.BUDGET.BUDGET_TYPE = "iterations"
_C.BUDGET.BUDGET_AMOUNT = 50


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
