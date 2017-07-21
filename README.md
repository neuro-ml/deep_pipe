# Deep pipe

Repository for deep learning experiments, primarily for 3d image segmentation


## Installation:
```
git clone --recursive https://github.com/neuro-ml/deep_pipe.git
```
Then you need to add the library to the list of python libraries. One of the ways:
```
ln -s /nmnt/media/home/USERNAME/deep_pipe/dpipe ~/env3.6/lib/python3.6/site-packages/ #(for virtualenv)
ln -s /nmnt/media/home/USERNAME/deep_pipe/dpipe ~/anaconda3/lib/python3.6/site-packages #(for conda)
```

## Simple usage

1. Create config file, examples are presented in `dpipe/scripts/config_examples`
2. Choose path to the experiment folder (it will be created for you). We will call it `EXPERIMENT_PATH`.
3. Run 
```
python scripts/experiment/build_experiment.py -cp CONFIG_PATH -ep EXPERIMENT_PATH -sp ..../deep_pipe/dpipe/scripts/simple
```
to make folder with experiment

4. Run 
```
python scripts/experiment/run_experiment_seq.py -ep EXPERIMENT_PATH
```

## Style guide:
- PEP8 style guide, which among other things limits max line length to 80 symbols.
- Also, it is forbidden to shadow python built-ins
