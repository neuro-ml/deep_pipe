from pathlib import Path

import pytest

from dpipe.config import Locker
from dpipe.layout.base import change_current_dir


def test_locker(tmpdir):
    locker = Locker(tmpdir)
    assert (tmpdir / '.lock').exists()
    with pytest.raises(FileExistsError):
        Locker(tmpdir)

    locker.run()
    assert not (tmpdir / '.lock').exists()


def test_chdir(tmpdir):
    with change_current_dir(tmpdir):
        locker = Locker()
        lock = Path('.lock')
        assert lock.exists()
        with pytest.raises(FileExistsError):
            Locker()

        locker.run()
        assert not lock.exists()

        def nested():
            assert lock.exists()
            with pytest.raises(FileExistsError):
                Locker()

        assert not lock.exists()
        Locker().run(nested())
        assert not lock.exists()


def test_no_instance():
    with pytest.raises(AttributeError):
        Locker.run()
