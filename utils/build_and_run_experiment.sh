~/deep_pipe/scripts/build_experiment.sh $1 $2
python ~/cluster-utils/scripts/run_experiment_cobrain.py $2 -c 1 -g 1 -r 50 -t 0 -pr -5 -pp ~/cluster-path
