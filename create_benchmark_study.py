import argparse
import os

from config import get_cfg_defaults
# FIXME: paths are somewhat broken but still work
# python main.py -vmp --config experiment_configs/iaml_xgboost_40981//RS_9.yaml
# replace // or make proper paths

parser = argparse.ArgumentParser(description="Specify benchmark to create")

parser.add_argument("-t", "--template", help="Path to template", dest="template")
parser.add_argument("-s", "--seed", help="Seed for DM", dest="seed", default=123)
parser.add_argument(
    "-r", "--replications", help="How many replications", dest="repls", default=10
)
parser.add_argument(
    "-f", "--seedrepls", help="Replications per seed", dest="seedrepls", default=5
)

args = parser.parse_args()

name = args.template.split("/")[-1].split(".")[-2]

experiment_directory = f"experiment_configs/{name}/"

#optimizers = ["BO", "EUBO", "qEUBO", "RS"]
optimizers = ["BO", "RS"]

cfg = get_cfg_defaults()
cfg.merge_from_file(args.template)

cfg["NAME"]["EXPERIMENT_NAME"] = name

if not os.path.exists(experiment_directory):
    os.mkdir(experiment_directory)

files = []


# Create Config files
for _ in range(args.seedrepls):
    for repl in range(args.repls):
        cfg["DECISION_MAKER"]["SEED"] = args.seed + repl + 1
        for optimizer in optimizers:
            cfg["OPTIMIZER"]["OPTIMIZER_TYPE"] = optimizer
            file = f"{experiment_directory}/{optimizer}_{repl+1}.yaml"
            files.append(file)
            with open(file, "+w") as f:
                f.writelines(cfg.dump())

# Create Run Script
script = f"experiment_scripts/run_{name}"
with open(script, "+w") as f:
    for file in files:
        f.write(f"python main.py -vmp --config {file}\n")

os.chmod(script, 0o775)
