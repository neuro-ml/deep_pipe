# Deep pipe

A collection of utils for deep learning experiments.


## Installation:
```bash
git clone https://github.com/neuro-ml/deep_pipe.git
cd deep_pipe
pip install -e .
```

## Documentation

https://deep-pipe.readthedocs.io/en/latest/

## Basic usage

1. Create a config file.
2. To build the experiment, run 
```bash
python -m dpipe build_experiment --config_path CONFIG_PATH --experiment_path EXPERIMENT_PATH
```

3. To start the experiment, run 
```bash
/path/to/deep_pipe/utils/run_experiment_seq.sh EXPERIMENT_PATH
```
