from dpipe.config import get_resource_manager, get_parser, parse_config

if __name__ == '__main__':
    parser = get_parser('config_path', 'train_ids_path', 'val_ids_path',
                        'log_path', 'save_model_path', 'restore_model_path')
    parser.add_argument(
        '--save', action='store_true', dest='save_on_quit',
        help='whether to save the model after ctrl+c is pressed'
    )

    resource_manager = get_resource_manager(parse_config(parser))

    save_on_quit = resource_manager['save_on_quit']
    save_model_path = resource_manager['save_model_path']
    model_controller = resource_manager['model_controller']

    model = resource_manager['model']
    train = resource_manager['train']

    with model_controller:
        try:
            train()
            model.save(save_model_path)
        except KeyboardInterrupt:
            if save_on_quit:
                model.save(save_model_path)
            else:
                raise
