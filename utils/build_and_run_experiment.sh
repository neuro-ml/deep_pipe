~/deep_pipe/scripts/build_experiment.sh --config_path $1 --experiment_path $2
python ~/cluster-utils/scripts/run_experiment_new.py $2 -c 1 -g 1 -r 50 -t 3 -pr -5 -pp ~/cluster-path
