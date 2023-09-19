import argparse
import os

from config import get_cfg_defaults

parser = argparse.ArgumentParser(description="Specify benchmark to create")

parser.add_argument("-t", "--template", help="Path to template", dest="template")
parser.add_argument("-s", "--seed", help="Seed for DM", dest="seed", default=123)
parser.add_argument(
    "-r",
    "--replications",
    help="How many replications (different seeds)",
    dest="repls",
    default=10,
)
parser.add_argument(
    "-f",
    "--seedrepls",
    help="How many replications per different seed",
    dest="seedrepls",
    default=10,
)

args = parser.parse_args()

name = args.template.split("/")[-1].split(".")[-2]

experiment_directory = f"experiment_configs/{name}"

optimizers = ["BO", "EUBO", "qEUBO", "RS"]

cfg = get_cfg_defaults()
cfg.merge_from_file(args.template)

cfg["NAME"]["EXPERIMENT_NAME"] = name

if not os.path.exists(experiment_directory):
    os.mkdir(experiment_directory)

files = []

# Create Config files
for seedrepl in range(args.seedrepls):
    cfg["SEEDREPL"] = seedrepl
    for repl in range(args.repls):
        cfg["DECISION_MAKER"]["SEED"] = args.seed + repl + 1
        for optimizer in optimizers:
            cfg["OPTIMIZER"]["OPTIMIZER_TYPE"] = optimizer
            file = f"{experiment_directory}/{optimizer}_{repl+1}_{seedrepl+1}.yaml"
            files.append(file)
            with open(file, "+w") as f:
                f.writelines(cfg.dump())

# Create Run Script
script = f"experiment_scripts/run_{name}"
with open(script, "+w") as f:
    for file in files:
        f.write(f"python main.py -v -s --config {file}\n")

os.chmod(script, 0o775)
