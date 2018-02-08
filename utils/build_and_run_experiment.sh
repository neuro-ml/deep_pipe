~/deep_pipe/scripts/build_experiment.sh --config_path $1 --experiment_path $2
python ~/cluster-utils/scripts/run_experiment_cobrain.py $2 -c 7 -g 1 -r 50 -pr 0 -pp "/home/krivov/cluster-path"
