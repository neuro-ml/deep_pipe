from dpipe.config import get_resource_manager, get_parser, parse_args

if __name__ == '__main__':
    parser = get_parser('config_path', 'train_ids_path', 'val_ids_path',
                        'log_path', 'save_model_path', 'restore_model_path')
    parser.add_argument(
        '--save', action='store_true', dest='save_on_quit',
        help='whether to save the model after ctrl+c is pressed'
    )

    rm = get_resource_manager(**parse_args(parser))

    model = rm.model
    if rm.restore_model_path is not None:
        model.load(rm.restore_model_path)
    try:
        rm.train()
        model.save(rm.save_model_path)
    except KeyboardInterrupt:
        if rm.save_on_quit:
            rm.model.save(rm.save_model_path)
        else:
            raise
