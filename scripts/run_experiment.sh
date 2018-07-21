# Simple script to run experiment.
# ./run_experiment.sh parameters == python do.py run_experiment --config_path args
python $(dirname $0)/do.py run_experiment --config_path "$@"
