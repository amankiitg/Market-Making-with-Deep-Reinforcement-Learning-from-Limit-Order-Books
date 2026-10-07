# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

from tensorflow import keras
from keras import backend as K

import tensorflow as tf

from state_spec import (AGENT_STATE_DIM, LOB_CHANNELS, MARKET_STATE_DIM, STATE_KEYS,
                        check_canonical_order, state_input_dims)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True                                #按需分配显存
K.set_session(tf.compat.v1.Session(config=config))

# Output width of the LOB encoder. Must stay in sync with make_compute_output_shape().
LOB_LATENT_DIM = 64


def get_lob_model(latent_dim, T):
    lob_state = keras.layers.Input(shape=(T, LOB_CHANNELS, 1), name='lob_state')

    conv_first1 = keras.layers.Conv2D(32, (1, 2), strides=(1, 2))(lob_state)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 5), strides=(1, 5))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    conv_first1 = keras.layers.Conv2D(32, (1, 4))(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)
    conv_first1 = keras.layers.Conv2D(32, (4, 1), padding='same')(conv_first1)
    conv_first1 = keras.layers.LeakyReLU(alpha=0.01)(conv_first1)

    # build the inception module
    convsecond_1 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)
    convsecond_1 = keras.layers.Conv2D(64, (3, 1), padding='same')(convsecond_1)
    convsecond_1 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_1)

    convsecond_2 = keras.layers.Conv2D(64, (1, 1), padding='same')(conv_first1)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)
    convsecond_2 = keras.layers.Conv2D(64, (5, 1), padding='same')(convsecond_2)
    convsecond_2 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_2)

    convsecond_3 = keras.layers.MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
    convsecond_3 = keras.layers.Conv2D(64, (1, 1), padding='same')(convsecond_3)
    convsecond_3 = keras.layers.LeakyReLU(alpha=0.01)(convsecond_3)

    convsecond_output = keras.layers.concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)
    conv_reshape = keras.layers.Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

    attn_input = conv_reshape
    attn_input_last = attn_input[:,-1:,:]  

    multi_head_attn_layer_1 = keras.layers.MultiHeadAttention(num_heads=10, key_dim=16, output_shape=latent_dim)

    attn_output, weight = multi_head_attn_layer_1(attn_input_last, attn_input, return_attention_scores=True)

    attn_output = keras.layers.Flatten()(attn_output)

    # add Batch Normalization
    # attn_output = keras.layers.BatchNormalization()(attn_output)

    # add Layer Normalization
    # attn_output = keras.layers.LayerNormalization()(attn_output)
    
    return keras.models.Model(lob_state, attn_output)


def get_fclob_model(latent_dim,T):
    print("This is the FC-LOB model")
    lob_state = keras.layers.Input(shape=(T, LOB_CHANNELS, 1), name='lob_state')

    dense_input = keras.layers.Flatten()(lob_state)

    # Hidden sizes follow the paper's FC-LOB description (1024, 256, 64) in section IV-B2.
    # Each layer must consume the previous one: the original code fed all three from
    # `dense_input`, which left the 1024 and 256 layers outside the graph entirely.
    dense_output = keras.layers.Dense(1024, activation='leaky_relu')(dense_input)
    dense_output = keras.layers.Dense(256, activation='leaky_relu')(dense_output)
    dense_output = keras.layers.Dense(latent_dim, activation='leaky_relu')(dense_output)

    return keras.models.Model(lob_state, dense_output)


def make_compute_output_shape(latent_dim):
    """Build the MultiHeadAttention output-shape workaround for a given encoder width.

    Keras (TensorFlow 2.6 era) cannot infer the output shape of the attention block, so
    `get_lob_model` is unusable as a layer without overriding `compute_output_shape`.
    Assign the result to the model instance, e.g.
    `lob_model.compute_output_shape = make_compute_output_shape(64)`.
    """

    def compute_output_shape(input_shape):
        return (input_shape[0], latent_dim)

    return compute_output_shape


def get_pretrain_model(model, T):
    lob_state = keras.layers.Input(shape=(T, LOB_CHANNELS, 1), name='lob_state')
    embedding = model(lob_state)
    output = keras.layers.Dense(3, activation='softmax')(embedding)

    return keras.models.Model(lob_state, output)


def get_model(lob_model, T, state_keys=None, market_state_dim=MARKET_STATE_DIM,
              agent_state_dim=AGENT_STATE_DIM):
    """Build the policy network (LOB embedding + raw market/agent features -> Dense head).

    The Keras input list is built by iterating `state_keys`, which defaults to the
    canonical `state_spec.STATE_KEYS` order. This MUST match the order in which the
    environments insert their states, because TensorForce calls the Keras model with
    `list(states.values())` (tensorforce/core/networks/keras.py, `KerasNetwork.apply`).

    Note on ordering vs wiring: an ordering mismatch between two same-shaped raw feature
    blocks (market_state and agent_state are both `(24,)`) that are concatenated and fed
    to a Dense layer only permutes the weight columns, so it does not change what the
    network can represent. It does change every weight matrix, so a pretrained checkpoint
    would be silently misaligned. Drift is prevented structurally here (single source of
    truth) and checked by tests/test_wiring.py.
    """
    keys = check_canonical_order(STATE_KEYS if state_keys is None else state_keys)
    dims = state_input_dims(
        T, keys=keys, market_state_dim=market_state_dim, agent_state_dim=agent_state_dim
    )

    input_ls = list()
    dense_input = list()
    for key in keys:
        state_input = keras.layers.Input(shape=dims[key], name=key)
        input_ls.append(state_input)
        if key == 'lob_state':
            dense_input.append(lob_model(state_input))
        else:
            dense_input.append(state_input)

    dense_input = keras.layers.concatenate(dense_input, axis=1)

    dense_output = keras.layers.Dense(64, activation='leaky_relu')(dense_input)

    return keras.models.Model(input_ls, dense_output)

if __name__ == '__main__':
    get_lob_model(64,50).summary()