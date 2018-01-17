# Simple script to print metrics across experiments
# Example ./print_aggregated_results mean ~/exp/brats_deepmedic/

$(dirname $0)/print_results.sh $1 $2 test_dices
