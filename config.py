from yacs.config import CfgNode as CN

_C = CN()

_C.NAME = CN()
_C.NAME.EXPERIMENT_NAME = "PBMOHPO"

_C.PROBLEM = CN()
_C.PROBLEM.PROBLEM_TYPE = "zdt1"

# ZDT1 Settings
_C.PROBLEM.DIMENSIONS = 10

# YAHPO Setting
_C.PROBLEM.ID = "lcbench"
_C.PROBLEM.INSTANCE = 3945
_C.PROBLEM.OBJECTIVE_NAMES = ["time", "val_accuracy"]
_C.PROBLEM.OBJECTIVE_SCALING_FACTORS = [1, 1]

# Fixed HPs for YAHPO
_C.FIXED_HPS = CN()

_C.FIXED_HPS.TRAINSIZE = (None, None)
_C.FIXED_HPS.EPOCH = (None, None)
_C.FIXED_HPS.REPL = (None, None)

_C.FIXED_HPS.REPLACE = (None, None)
_C.FIXED_HPS.RESPECT_UNORDERED_FACTORS = (None, None)
_C.FIXED_HPS.SPLITRULE = (None, None)

_C.FIXED_HPS.BOOSTER = (None, None)

_C.FIXED_HPS.DEFINITION = (None, None)

_C.FIXED_HPS.PRE_POST = (None, None)

_C.FIXED_HPS.NUM_IMPUTE_SELECTED_CPO = (None, None)

_C.PROBLEM.OML_TASK = 359960

_C.OPTIMIZER = CN()
_C.OPTIMIZER.OPTIMIZER_TYPE = "BO"

_C.DECISION_MAKER = CN()
_C.DECISION_MAKER.DECISION_MAKER_TYPE = "DecisionMaker"
_C.DECISION_MAKER.SEED = 123

_C.BUDGET = CN()
_C.BUDGET.EVAL_BUDGET = 50
_C.BUDGET.DM_BUDGET = 50

_C.BATCH_SIZE = CN()
_C.BATCH_SIZE.EVAL_BATCH_SIZE = 1
_C.BATCH_SIZE.DM_BATCH_SIZE = 1

_C.SEEDREPL = 0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
