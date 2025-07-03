from functools import partial
from typing import Optional, Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import numpy as jnp
from optax import ScaleByAdamState, GradientTransformation, ScalarOrSchedule, scale_by_learning_rate
from optax import tree_utils as otu
from optax._src import numerics, utils


def l2_regularization(params: dict, alpha: float = 0.001):
    return sum(
        alpha * (w ** 2).mean()
        for w in jax.tree_leaves(params["params"])
    )


@partial(jax.jit, inline=True)
def tree_bias_correction(moment, decay, count):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    def bias_correction(t, c):
        bias_correction_ = 1 - decay**c
        return t / bias_correction_.astype(t.dtype)

    # Perform division in the original precision.
    return jax.tree.map(
        bias_correction, moment, count)


def scale_adam_with_array_counts(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        mu_dtype: Optional[chex.ArrayDType] = None,
        *,
        nesterov: bool = False,
) -> GradientTransformation:
    r"""Rescale updates according to the Adam algorithm, except
    counts are kept per parameter.

    See :func:`optax.adam` for more details.

    Args:
      b1: Decay rate for the exponentially weighted average of grads.
      b2: Decay rate for the exponentially weighted average of squared grads.
      eps: Term added to the denominator to improve numerical stability.
      eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
      mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
      nesterov: Whether to use Nesterov momentum. The variant of Adam with
        Nesterov momentum is described in [Dozat 2016]

    Returns:
      A :class:`optax.GradientTransformation` object.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        count = otu.tree_zeros_like(params, dtype=jnp.int32)
        return ScaleByAdamState(count=count, mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = jax.tree.map(lambda c: numerics.safe_int32_increment(c), state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: b1 * m + (1 - b1) * g,
                tree_bias_correction(mu, b1, numerics.safe_int32_increment(count_inc)),
                tree_bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = tree_bias_correction(mu, b1, count_inc)
        # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
        # unclear why. Other Nadam implementations also omit the extra b2 factor.
        nu_hat = tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: None if m is None else m / (jnp.sqrt(v + eps_root) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = otu.tree_cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


def adam_with_param_counts(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
):
    return optax.chain(
      scale_adam_with_array_counts(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      scale_by_learning_rate(learning_rate),
  )


def numpyify(leaf):
    if isinstance(leaf, jnp.ndarray):
        return np.array(leaf)
    return leaf