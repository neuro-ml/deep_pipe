# Simple script to print metrics across experiments
# Example ./print_results mean test_metrics/dice.json ~/exp/brats_deepmedic/

UTILS_PATH=$(dirname $0)

RESULT_FILES=$(ls $3/experiment_*/$2 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "No results found"
    exit
fi

echo $(echo ${RESULT_FILES} | wc -w) experiment resuts found
${UTILS_PATH}/jq -s add ${RESULT_FILES} | python ${UTILS_PATH}/aggregate.py $1