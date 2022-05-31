import os
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import wandb
from dpipe.commands import load_from_folder
from dpipe.im.utils import zip_equal
from dpipe.io import PathLike
from wandb.sdk.wandb_run import Run as wandbRun

__all__ = 'Logger', 'ConsoleLogger', 'TBLogger', 'NamedTBLogger', 'WANDBLogger'


def log_vector(logger, tag: str, vector, step: int):
    for i, value in enumerate(vector):
        logger.log_scalar(tag=tag + f'/{i}', value=value, step=step)


def log_scalar_or_vector(logger, tag, value: np.ndarray, step):
    value = np.asarray(value).flatten()
    if value.size > 1:
        log_vector(logger, tag, value, step)
    else:
        logger.log_scalar(tag, value, step)


def make_log_vector(logger, tag: str, first_step: int = 0) -> Callable:
    def log(tag, value, step):
        log_vector(logger, tag, value, step)

    return logger._make_log(tag, first_step, log)


def group_dicts(dicts):
    groups = defaultdict(list)
    for entry in dicts:
        for name, value in entry.items():
            groups[name].append(value)

    return dict(groups)


class Logger:
    """Interface for logging during training."""

    def _dict(self, prefix, d, step):
        for name, value in d.items():
            self.value(f'{prefix}{name}', value, step)

    def train(self, train_losses: Sequence, step: int):
        """Log the ``train_losses`` at current ``step``."""
        raise NotImplementedError

    def value(self, name: str, value, step: int):
        """Log a single ``value``."""
        raise NotImplementedError

    def policies(self, policies: dict, step: int):
        """Log values coming from `ValuePolicy` objects."""
        self._dict('policies/', policies, step)

    def metrics(self, metrics: dict, step: int):
        """Log the metrics returned by the validation function during training."""
        self._dict('val/metrics/', metrics, step)


class ConsoleLogger(Logger):
    """A logger that writes to to stdout."""

    def value(self, name, value, step):
        print(f'{step:>05}: {name}: {value}', flush=True)

    def train(self, train_losses: Sequence[Union[dict, tuple, float]], step):
        text = ''
        if train_losses and isinstance(train_losses[0], dict):
            for name, values in group_dicts(train_losses).items():
                text += f'{name}: {np.mean(values)} '

        else:
            text += str(np.mean(train_losses, axis=0))

        self.value('Train loss', text, step)

    def policies(self, policies: dict, step: int):
        self._dict('Policies: ', policies, step)

    def metrics(self, metrics: dict, step: int):
        self._dict('Metrics: ', metrics, step)


class TBLogger(Logger):
    """A logger that writes to a tensorboard log file located at ``log_path``."""

    def __init__(self, log_path: PathLike):
        import tensorboard_easy
        self.logger = tensorboard_easy.Logger(log_path)

    def train(self, train_losses: Sequence[Union[dict, tuple, float]], step):
        if train_losses and isinstance(train_losses[0], dict):
            for name, values in group_dicts(train_losses).items():
                self.value(f'train/loss/{name}', np.mean(values), step)

        else:
            log_scalar_or_vector(self.logger, 'train/loss', np.mean(train_losses, axis=0), step)

    def value(self, name, value, step):
        dirname, base = os.path.split(name)

        count = base.count('__')
        if count > 1:
            raise ValueError(f'The tag name must contain at most one magic delimiter (__): {base}.')
        if count:
            base, kind = base.split('__')
        else:
            kind = 'scalar'

        name = os.path.join(dirname, base)

        if kind == 'vector':
            log = partial(log_vector, self.logger)
        else:
            log = getattr(self.logger, f'log_{kind}')
        log(name, np.asarray(value), step)

    def __getattr__(self, item):
        return getattr(self.logger, item)


class NamedTBLogger(TBLogger):
    """
    A logger that writes multiple train losses to a tensorboard log file located at ``log_path``.

    Each loss is assigned to a corresponding tag name from ``loss_names``.
    """

    def __init__(self, log_path: PathLike, loss_names: Sequence[str]):
        super().__init__(log_path)
        self.loss_names = loss_names

    def train(self, train_losses, step):
        values = np.mean(train_losses, axis=0)
        for name, value in zip_equal(self.loss_names, values):
            self.logger.log_scalar(f'train/loss/{name}', value, step)


class WANDBLogger(Logger):
    def __init__(
        self,
        project: Optional[str],
        run_name: Optional[str] = None,
        *,
        group: Optional[str] = None,
        entity: str = 'neuro-ml',
        config: Union[Dict, str, None] = None,
        dir: Optional[str] = None,
        resume: str = 'auto',
        **watch_kwargs: Any,
    ) -> None:
        """A logger that writes to a wandb run.

        Call `wandb login` before first usage.
        """

        settings = [None, wandb.Settings(start_method='fork'), wandb.Settings(start_method='thread')]
        exp = None
        for i, s in enumerate(settings):
            try:
                exp = wandb.init(
                    entity=entity, project=project, resume=resume, group=group, dir=dir,
                    settings=s
                )
                break
            except wandb.errors.UsageError:
                warnings.warn(f"Couldn't init wandb with setting {i}, trying another one.")
                continue

        assert isinstance(exp, wandbRun), 'Failed to register launch with wandb'

        current_fold_root = Path(exp.dir).parent.parent.parent
        experiment_root = current_fold_root.parent

        # find out if the experiment is cut into several folds
        cut_into_folds = (
            len(
                [
                    p
                    for p in experiment_root.glob('*')
                    if p.name.startswith('experiment_') and p.is_dir()
                ]
            )
            > 1
        )

        current_experiment_number = (
            str(int(current_fold_root.name.replace('experiment_', '')))
            if cut_into_folds
            else 0
        )

        if run_name is not None:
            exp.name = run_name  # can be changed manually
        else:
            name = experiment_root.name
            if cut_into_folds:
                name = f'{name}-{current_experiment_number}'
            exp.name = name
        artifact = wandb.Artifact('model', type='config')

        try:
            artifact.add_file(
                str(experiment_root / 'resources.config'), f'{exp.name}/config.txt'
            )
            # all json files of the current fold are added as artifacts
            for json in current_fold_root.glob('*.json'):
                artifact.add_file(str(json), f'{exp.name}/{json.name}')
        except ValueError:
            warnings.warn("It's likely you don't run a usual experiment, some artifacts were not found")

        self._experiment = exp

        wandb.log_artifact(artifact)

        self.update_config(dict(experiment=experiment_root.name))
        if cut_into_folds:
            self.update_config(dict(fold=current_experiment_number))
        if config is not None:
            self.update_config(config)

        if watch_kwargs:
            self.watch(**watch_kwargs)

    def __del__(self):
        wandb.finish()

    @property
    def experiment(self) -> wandbRun:
        return self._experiment

    def value(self, name: str, value: Any, step: Optional[int] = None) -> None:
        self._experiment.log({name: value, 'epoch': step})

    def train(
        self, train_losses: Union[Sequence[Dict], Sequence[float], Sequence[tuple], Sequence[np.ndarray]], step: int
    ) -> None:
        if not train_losses:
            return None
        train_losses_types = {type(tl) for tl in train_losses}
        assert len(train_losses_types) == 1, 'Inconsistent train_losses'
        t = train_losses_types.pop()
        if issubclass(t, dict):
            for name, values in group_dicts(train_losses).items():
                self.value(f'train/loss/{name}', np.mean(values), step)
        elif issubclass(t, (float, tuple, np.ndarray)):
            self.value('train/loss', np.mean(train_losses), step)
        else:
            msg = f'The elements of the train_losses are expected to be of dict, float, tuple or numpy array type, but the elements are of {t.__name__} type'
            raise NotImplementedError(msg)

    def watch(self, **kwargs) -> None:
        self.experiment.watch(**kwargs)

    def update_config(self, config_args) -> None:
        self.experiment.config.update(config_args, allow_val_change=True)

    def agg_metrics(
        self, agg_metrics: Union[dict, str, Path], section: str = ''
    ) -> None:
        """Log final metrics calculated in the end of experiment to summary table.
        Idea is to use these values for preparing leaderboard.

        agg_metrics: dictionary with name of metric as a key and with its value
        """
        if isinstance(agg_metrics, str) or isinstance(agg_metrics, Path):
            agg_metrics = {
                k if not section else f'{section}/{k}': v
                for k, v in load_from_folder(agg_metrics, ext='.json')
            }
        elif section:
            agg_metrics = {f'{section}/{k}': v for k, v in agg_metrics.items()}

        for k, v in agg_metrics.items():
            self.experiment.summary[k] = v
            # self.experiment.summary.update()

    def ind_metrics(self, ind_metrics: Any, step: int = 0, section: Optional[str] = None) -> None:
        """Save individual metrics to a table to see bad cases

        ind_metrics: DataFrame
        step: int
        section: str, defines some metrics' grouping
        """
        import pandas as pd
        from wandb import Table

        if isinstance(ind_metrics, str) or isinstance(ind_metrics, Path):
            ind_metrics = pd.DataFrame.from_dict(
                {k: v for k, v in load_from_folder(ind_metrics, ext='.json')}
            ).reset_index().round(2)
        table = Table(dataframe=ind_metrics)

        name = (
            'Individual Metrics' if section is None else f'{section}/Individual Metrics'
        )
        self.experiment.log({name: table})

    def image(
        self,
        name: str,
        *values,
        step: int,
        section: Optional[str] = None,
        masks_keys: tuple = ('predictions', 'ground_truth'),
    ) -> None:
        """Method that logs images (set by values),
        each value is a dict with fields, preds, target and optinally caption defined
        Special policy that works as callback
        """
        from wandb import Image

        name = name if section is None else f'{section}/{name}'
        self.experiment.log(
            {
                name: [
                    Image(
                        value['image'],
                        masks={k: {'mask_data': value[k]} for k in masks_keys},
                        caption=value.get('caption', None),
                    )
                    for value in values
                ],
            },
            step=step,
        )

    def log_info(self, name: str, wandb_converter, *infos, section: Optional[str] = None, step: Optional[int] = None) -> None:
        name = name if section is None else f'{section}/{name}'
        self.experiment.log({name: [wandb_converter(info) for info in infos]})
