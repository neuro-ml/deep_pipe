import os
from dpipe.config import get_paths

rules_path = get_paths()['rules']

include: os.path.join(rules_path, "preamble.smk")
include: os.path.join(rules_path, "core.smk")
include: os.path.join(rules_path, "evaluate_individual.smk")
include: os.path.join(rules_path, "gifs.smk")
include: os.path.join(rules_path, "precision_recall_curve.smk")
