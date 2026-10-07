"""TensorForce 0.6.5 compatibility shims.

Why this file exists
--------------------
``tensorforce/core/optimizers/tf_optimizer.py`` initialises a Keras optimizer like this::

    try:
        self.tf_optimizer._create_all_weights(var_list=variables)
    except AttributeError:
        self.tf_optimizer._create_hypers()
        self.tf_optimizer._create_slots(var_list=variables)

``_create_all_weights`` / ``_create_hypers`` / ``_create_slots`` are private Keras
optimizer internals. Keras 2.11 replaced them with the public ``Optimizer.build(var_list)``,
so on TensorFlow >= 2.11 the ``try`` raises ``AttributeError`` and the fallback raises again::

    AttributeError: 'Adam' object has no attribute '_create_hypers'

TensorForce 0.6.5 was authored against TensorFlow 2.6 (its own metadata pins
``tensorflow==2.6.0``), where the private methods still existed.

What is audited
---------------
Every private Keras optimizer API used by TensorForce 0.6.5 was grepped:

* ``_create_all_weights``, ``_create_hypers``, ``_create_slots``: only in
  ``TFOptimizer.initialize_given_variables`` (handled by this shim).
* ``get_slot``, ``_resource_apply_dense``, ``_resource_apply_sparse``, ``_prepare``,
  ``_distributed_apply``: not used anywhere in the package.
* ``TFOptimizer.step`` (the code path exercised by ``agent.update()``) only calls the
  public ``self.tf_optimizer.apply_gradients(grads_and_vars=...)``, which is stable
  across TensorFlow 2.6 to 2.16.

So patching ``initialize_given_variables`` is sufficient for the whole
act / experience / update loop. The shim is a no-op when the private methods exist
(TensorFlow <= 2.10), detected by feature probe rather than by parsing a version string.
"""

import logging

import tensorflow as tf

LOGGER = logging.getLogger(__name__)

_SHIM_STATE = {'applied': None}


def _optimizer_uses_private_creation_api():
    """True if this TensorFlow/Keras still exposes the private optimizer API TensorForce calls."""
    probe = tf.keras.optimizers.Adam(learning_rate=1e-3)
    return hasattr(probe, '_create_all_weights') and hasattr(probe, '_create_hypers')


def apply_tensorforce_tf_compat(verbose=True):
    """Install the optimizer shim if the running TensorFlow needs it.

    Returns True if the shim was installed by this call, False if it was not needed or was
    already installed. Safe to call more than once.
    """
    if _SHIM_STATE['applied'] is not None:
        return False

    if _optimizer_uses_private_creation_api():
        _SHIM_STATE['applied'] = False
        if verbose:
            LOGGER.info(
                'TensorForce/TF compatibility shim not needed: '
                'tensorflow %s still provides the private optimizer API.',
                tf.__version__,
            )
        return False

    from tensorforce.core.optimizers.optimizer import Optimizer
    from tensorforce.core.optimizers.tf_optimizer import TFOptimizer

    def initialize_given_variables(self, *, variables):
        # Same call as the original `super().initialize_given_variables(...)`: TFOptimizer
        # inherits from Optimizer only, so this is exactly the original statement.
        Optimizer.initialize_given_variables(self, variables=variables)

        optimizer = self.tf_optimizer
        if hasattr(optimizer, '_create_all_weights'):
            optimizer._create_all_weights(var_list=variables)
        else:
            # Public replacement for _create_hypers() + _create_slots() in Keras >= 2.11.
            optimizer.build(variables)

    TFOptimizer.initialize_given_variables = initialize_given_variables
    _SHIM_STATE['applied'] = True
    if verbose:
        LOGGER.warning(
            'Installed TensorForce/TF compatibility shim for tensorflow %s '
            '(Keras optimizer internals _create_all_weights/_create_hypers are gone). '
            'See compat.py for the audit of the patched API surface.',
            tf.__version__,
        )
    return True


def shim_status():
    """Return the shim state: True (installed), False (not needed) or None (not probed yet)."""
    return _SHIM_STATE['applied']
