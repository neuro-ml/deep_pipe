# Simple script to build experiment.
# ./build_experiments.sh parameters == python do.py build_experiment parameters
python $(dirname $0)/do.py build_experiment "$@"
