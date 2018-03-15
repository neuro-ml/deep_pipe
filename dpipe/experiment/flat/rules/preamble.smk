import os

from dpipe.config import get_paths


DO = get_paths()['do']
RULES_PATH = get_paths()['rules']

CONFIG_ARG = '--config_path ../config'
SAVED_MODEL = 'model'
TRAINING_LOG = 'train_logs'

SETS = ['train', 'val', 'test']
TRAIN_IDS, VAL_IDS, TEST_IDS = expand('{sets}_ids.json', sets=SETS)
