"""Pretrain the Attn-LOB encoder as a 3-class mid-price trend classifier.

Why this file exists: `main.py` loads `./ckpt/pretrain_model_{code}/weights` unless
`--wo_pretrain=True`, and the original repository shipped neither the checkpoint nor the
script that produces it. The helpers (`utils.getLabel`, `utils.onehot_label`,
`utils.data_classification`, `utils.load_data`) were already present but unused.

Paper reference (arXiv:2305.15821, section IV-B1 and section III-B):
  * label: 3 classes, up / stationary / down, from the mid-price trend;
  * trend horizon k = 10 events and threshold alpha = 1e-5 (values are stated);
  * state window T = 50 (stated);
  * the pretrained network is then reused as the encoder inside the RL agent
    (`main.py:build_lob_encoder` takes `model_pretrain.layers[1]`).

Values marked `# ASSUMED` are not stated in the paper; they were picked to be reasonable
and are exposed as CLI flags so they can be swept.

Known fidelity gap between this pretraining path and the RL state path:
  `utils.process_data` normalises volume levels by the maximum over the whole loaded
  dataset, while `utils.lob_norm` (used by the environments) normalises by the maximum
  inside the T-row window. Price channels agree, because both divide by the row's own mid
  price. Volume channels therefore have a different scale at pretraining time. Fixing it
  properly means choosing one normalisation constant for both paths; that is deliberately
  left to the caller because it changes the meaning of the pretrained weights.

Usage:
    python -m data.adapter --synthetic --code 000001 --day 20191101 --rows 4000
    python pretrain.py --code 000001 --days 20191101
    # then, without --wo_pretrain:
    python main.py --code 000001 --train-days 20191101 --test-days 20191101
"""

import os
import random
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np
import pandas as pd
import pyrallis
from tensorflow import keras

from network.network import (LOB_LATENT_DIM, get_lob_model, get_pretrain_model,
                             make_compute_output_shape)
from state_spec import LOB_COLUMNS
from utils import data_classification, load_data, onehot_label, process_data


@dataclass
class PretrainConfig:
    # Data
    code: str = '000001'
    days: List[str] = field(default_factory=lambda: [
        '20191101', '20191104', '20191105', '20191106', '20191107', '20191108',
        '20191111', '20191112'])
    data_dir: str = './data'
    out_dir: str = './ckpt'
    # Model / labels, paper section IV-B1
    time_window: int = 50           # T
    horizon: int = 10               # k
    threshold: float = 1e-5         # alpha
    latent_dim: int = LOB_LATENT_DIM
    # Training, not stated in the paper
    epochs: int = 20                # ASSUMED
    batch_size: int = 128           # ASSUMED
    learning_rate: float = 1e-3     # ASSUMED
    validation_split: float = 0.1   # ASSUMED
    seed: int = 0


def build_dataset(config):
    """Return (X, Y) ready for a softmax classifier over 3 trend classes."""
    data = load_data(config['code'], config['days'], horizon=config['horizon'],
                     data_dir=config['data_dir'])
    data = process_data(data)

    # Feature order must match what the environment feeds the encoder, see
    # state_spec.LOB_COLUMNS. process_data keeps the raw column order, so select explicitly
    # rather than relying on it. The price columns from price.csv are not part of the
    # paper's LOB input (Eq. 1) and are excluded on purpose.
    missing = [column for column in LOB_COLUMNS if column not in data.columns]
    if missing:
        raise KeyError(f'loaded data is missing LOB column(s): {missing[:5]}')
    if len(data) <= config['time_window']:
        raise ValueError(
            f'only {len(data)} rows left after utils.process_data. It keeps rows with '
            f"10:00:00 < time < 14:30:00 and drops the last {config['horizon']} rows with an "
            f"undefined label, so the dataset must cover the mid-session window."
        )

    labels = data['y'].to_numpy()
    distribution = pd.Series(labels).value_counts(normalize=True).sort_index()
    print('label distribution (1=up, 2=stationary, 3=down):')
    print(distribution.to_string())

    X, Y = data_classification(data[list(LOB_COLUMNS)].to_numpy(), labels,
                               config['time_window'])
    Y = onehot_label(pd.DataFrame(Y))
    return X, Y


@pyrallis.wrap()
def main(config: PretrainConfig):
    config = asdict(config)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    if hasattr(keras.utils, 'set_random_seed'):
        keras.utils.set_random_seed(config['seed'])

    X, Y = build_dataset(config)
    print(f'pretraining samples: X {X.shape}, Y {Y.shape}')

    encoder = get_lob_model(config['latent_dim'], config['time_window'])
    # Keras cannot infer the MultiHeadAttention output shape, see network module.
    encoder.compute_output_shape = make_compute_output_shape(config['latent_dim'])
    model = get_pretrain_model(encoder, config['time_window'])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.fit(
        X, Y,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        verbose=2,
    )

    out_dir = os.path.join(config['out_dir'], 'pretrain_model_' + config['code'])
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(out_dir, 'weights')
    model.save_weights(checkpoint_filepath)
    print('Encoder weights saved to', checkpoint_filepath)
    print('main.py will load them unless --wo_pretrain=True is passed.')


if __name__ == '__main__':
    main()
