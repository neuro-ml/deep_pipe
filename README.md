# Deep pipe

Repository for deep learning experiments with 3d image segmentation


## Installation:
```
git clone https://github.com/neuro-ml/deep_pipe.git

```
Then you need to add the library to the list of python libraries. One of the ways:
```
ln -s /path/to/deep_pipe/dpipe /path/to/virtualenv/lib/python3.6/site-packages/ # for virtualenv
# or
ln -s /path/to/deep_pipe/dpipe /path/to/anaconda/lib/python3.6/site-packages/ # for conda
```

## Basic usage

1. Create a config file. There are some examples in `config_examples`
2. To build the experiment, run 
```
python /path/to/deep_pipe/scripts/do.py build_experiment --config_path CONFIG_PATH --experiment_path EXPERIMENT_PATH
```

3. To start the experiment, run 
```
/path/to/deep_pipe/experiment/run_experiment_seq.sh EXPERIMENT_PATH
```

## Requirements

Only Python 3.6 is supported for now.
Other requirements are listed in `requirements.txt`.
