"""Shared fixtures. Run from the repository root: `pytest`."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

CODE = '000001'
DAY = '20191101'
TIME_WINDOW = 50


@pytest.fixture(scope='session')
def dataset(tmp_path_factory):
    """A schema-valid synthetic dataset, see data/adapter.py."""
    from data.adapter import generate_synthetic

    root = tmp_path_factory.mktemp('dataset')
    data_root = str(root / 'data')
    raw_root = str(root / 'raw')
    generate_synthetic(
        data_root=data_root, raw_root=raw_root, code=CODE, day=DAY,
        rows=400, trade_stride=4, seed=7)
    return dict(data_root=data_root, raw_root=raw_root, code=CODE, day=DAY)


@pytest.fixture(scope='session')
def sparse_dataset(tmp_path_factory):
    """A dataset with a single trade, to exercise the unusable-episode guard."""
    from data.adapter import generate_synthetic

    root = tmp_path_factory.mktemp('sparse_dataset')
    data_root = str(root / 'data')
    raw_root = str(root / 'raw')
    generate_synthetic(
        data_root=data_root, raw_root=raw_root, code=CODE, day=DAY,
        rows=400, trade_stride=10 ** 6, seed=7)
    return dict(data_root=data_root, raw_root=raw_root, code=CODE, day=DAY)


def make_env(dataset, env_type='continuous', **overrides):
    from environment.env_continuous import EnvContinuous
    from environment.env_discrete import EnvDiscrete

    kwargs = dict(
        code=dataset['code'],
        day=dataset['day'],
        latency=1,
        T=TIME_WINDOW,
        log=0,
        experiment_name='pytest',
        data_dir=dataset['data_root'],
        raw_dir=dataset['raw_root'],
    )
    kwargs.update(overrides)
    if env_type == 'continuous':
        return EnvContinuous(**kwargs)
    return EnvDiscrete(**kwargs)


@pytest.fixture
def env_continuous(dataset):
    return make_env(dataset, env_type='continuous')


@pytest.fixture
def env_discrete(dataset):
    return make_env(dataset, env_type='discrete')
