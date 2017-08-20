from dpipe.config import get_config, get_resource_manager

if __name__ == '__main__':
    config = get_config('config_path', 'experiment_path')
    get_resource_manager(config)['experiment']
    print('Experiment was build')
