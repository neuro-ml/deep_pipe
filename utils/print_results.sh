# Simple script to print metrics across experiments
# Example ./print_results mean ~/exp/brats_deepmedic/ test_dices

UTILS_PATH=$(dirname $0)
DICES_FILES="$2/experiment_*/$3.json"

N_RESULTS=$(echo $DICES_FILES | wc -w)
echo Experiment resuts found: $N_RESULTS
$UTILS_PATH/jq -s add $2/experiment_*/test_dices.json | python $UTILS_PATH/aggregate.py $1

