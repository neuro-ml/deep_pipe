"""Script that runs command from config."""
from warnings import warn

from dpipe.config.base import render_config_resource

warn('"python do.py" is deprecated. Use "python -m dpipe" instead.', DeprecationWarning)

if __name__ == '__main__':
    render_config_resource()
