import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple

def compute_matrix_rank(
    m: jnp.ndarray, 
    prop: float = 0.99
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes (rank, effective_rank, approximate_rank, approximate_rank_abs) for a 2D matrix m.
    """
    # 1) Singular values
    sv = jnp.linalg.svd(m, compute_uv=False)
    
    # 2) Exact rank
    rank = jnp.count_nonzero(sv).astype(jnp.int32)
    
    # 3) Effective rank via Shannon entropy
    norm_sv = sv / jnp.sum(jnp.abs(sv))
    entropy = -jnp.sum(jnp.where(norm_sv > 0, norm_sv * jnp.log(norm_sv), 0.0))
    effective_rank = jnp.exp(entropy).astype(jnp.float32)
    
    # 4) Approximate rank (based on squared singular values)
    sq_sv = sv**2
    norm_sq = sq_sv / jnp.sum(sq_sv)
    desc_sq = jnp.sort(norm_sq)[::-1]
    cumsum_sq = jnp.cumsum(desc_sq)
    approx_rank = (jnp.argmax(cumsum_sq >= prop) + 1).astype(jnp.int32)
    
    # 5) Approximate rank (based on absolute singular values)
    norm_abs = sv / jnp.sum(sv)
    desc_abs = jnp.sort(norm_abs)[::-1]
    cumsum_abs = jnp.cumsum(desc_abs)
    approx_rank_abs = (jnp.argmax(cumsum_abs >= prop) + 1).astype(jnp.int32)
    
    return rank, effective_rank, approx_rank, approx_rank_abs

batched_summaries = jax.vmap(
    lambda m: compute_matrix_rank(m, prop=0.99),
    in_axes=0, out_axes=(0, 0, 0, 0)
)

def summarize_all_layers(
    ms: jnp.ndarray,      # shape (num_layers, rows, cols)
    prop: float = 0.99
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For a batch of weight matrices `ms`, returns five arrays each of shape (num_layers,):
    1) exact ranks
    2) effective ranks
    3) approx ranks (sq)
    4) approx ranks (abs)
    5) dead neuron counts (columns that sum to zero)
    """
    ms = jnp.asarray(ms)
    ranks, eff_ranks, approx_ranks, approx_ranks_abs = batched_summaries(ms)
    
    # Dead‐neuron count: for each layer, sum abs over rows => per‐column sums, count zeros
    col_sums = jnp.sum(jnp.abs(ms), axis=1)      # shape (num_layers, cols)
    dead_neurons = jnp.sum(col_sums == 0, axis=1).astype(jnp.int32)
    
    return ranks, eff_ranks, approx_ranks, approx_ranks_abs, dead_neurons