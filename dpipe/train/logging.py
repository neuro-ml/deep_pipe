from tensorboard_easy import Logger


def log_vector(logger: Logger, tag: str, vector, step: int):
    """Adds a vector values to log."""
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def make_log_vector(logger: Logger, tag: str, first_step: int = 0) -> callable:
    def log(tag, value, step):
        log_vector(logger, tag, value, step)

    return logger._make_log(tag, first_step, log)
