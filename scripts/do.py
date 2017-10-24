from dpipe.config import get_args, get_resource_manager

if __name__ == '__main__':
    get_resource_manager(get_args('config_path', 'experiment_path')).experiment
print('The experiment is built')