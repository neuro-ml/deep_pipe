# Simple script to print metrics across experiments
# Example ./print_results mean ~/exp/brats_deepmedic/ test_dices

set -e

UTILS_PATH=$(dirname $0)

RESULT_FILES=$(find $2/experiment_* -maxdepth 1 -name $3.json)

echo $(echo ${RESULT_FILES} | wc -w) experiment resuts found
${UTILS_PATH}/jq -s add ${RESULT_FILES} | python ${UTILS_PATH}/aggregate.py $1