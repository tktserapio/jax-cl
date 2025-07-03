# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for running the Lanczos algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as jnp
import jax.random as random


# TODO(gilmer) This function should use higher numerical precision?
def lanczos_alg(matrix_vector_product, dim, order, rng_key):
  """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

  This function applies Lanczos algorithm of a given order.  This function
  does full reorthogonalization.

  WARNING: This function may take a long time to jit compile (e.g. ~3min for
  order 90 and dim 1e7).

  Args:
    matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
      Input/Output must be of shape [dim].
    dim: Matrix H is [dim, dim].
    order: An integer corresponding to the number of Lanczos steps to take.
    rng_key: The jax PRNG key.

  Returns:
    tridiag: A tridiagonal matrix of size (order, order).
    vecs: A numpy array of size (order, dim) corresponding to the Lanczos
      vectors.
  """

  tridiag = jnp.zeros((order, order))
  vecs = jnp.zeros((order, dim))

  init_vec = random.normal(rng_key, shape=(dim,))
  init_vec = init_vec / jnp.linalg.norm(init_vec)
  vecs = vecs.at[0].set(init_vec)

  beta = 0
  # TODO(gilmer): Better to use lax.fori loop for faster compile?
  for i in range(order):
    v = vecs[i, :].reshape((dim))
    if i == 0:
      v_old = 0
    else:
      v_old = vecs[i - 1, :].reshape((dim))

    w = matrix_vector_product(v)
    assert (w.shape[0] == dim and len(w.shape) == 1), (
        'Output of matrix_vector_product(v) must be of shape [dim].')
    w = w - beta * v_old

    alpha = jnp.dot(w, v)
    tridiag = tridiag.at[i, i].set(alpha)
    w = w - alpha * v

    # Full Reorthogonalization
    for j in range(i):
      tau = vecs[j, :].reshape((dim))
      coeff = jnp.dot(w, tau)
      w += -coeff * tau

    beta = jnp.linalg.norm(w)
    # TODO(gilmer): The tf implementation raises an exception if beta < 1e-6
    # here. However JAX cannot compile a function that has an if statement
    # that depends on a dynamic variable. Should we still handle this base?
    # beta being small indicates that the lanczos vectors are linearly
    # dependent.

    if i + 1 < order:
      tridiag = tridiag.at[i,   i+1].set(beta)
      tridiag = tridiag.at[i+1, i  ].set(beta)
      vecs = vecs.at[i+1].set(w / beta)
  return (tridiag, vecs)