from functools import partial

import chex
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from optax import ScaleByAdamState, EmptyState


def num_top_mask(mask, vals, num_top):
    """
    Calculates the number of eligible elements, then
    the top eligible elements among the eligible elements in vals.
    """
    # Flatten the mask and vals arrays
    # jax.debug.print("num_top:{num_top}", num_top=num_top)
    mask_flat = mask.flatten()
    # jax.debug.print("mask: {mask}", mask=mask_flat.shape)
    vals_flat = vals.flatten()

    # Apply the mask to get values where mask is True
    masked_vals = jnp.where(mask_flat, vals_flat, -jnp.inf)

    # Calculate the number of top elements to select
    # Get the threshold value for the top p% elements
    threshold = jnp.sort(masked_vals)[num_top]

    # Create a new mask for elements that are above or equal to the threshold
    top_p_mask_flat = (masked_vals <= threshold) & (mask_flat)
    # Reshape back to the original shape
    return top_p_mask_flat.reshape(mask.shape)


def generate_seeds_for_pytree(key, pytree):
    """
    Generate a unique PRNGKey for each leaf in a PyTree.

    Parameters:
    - key: JAX PRNGKey to start generating unique subkeys.
    - pytree: The PyTree for which to generate unique subkeys.

    Returns:
    - A PyTree of the same structure as `pytree`, where each leaf is a unique PRNGKey.
    """
    # Flatten the PyTree and count the leaves
    leaves, treedef = jax.tree.flatten(pytree)

    # Split the original key into as many subkeys as there are leaves
    subkeys = jax.random.split(key, num=len(leaves))

    # Reconstruct the PyTree with the subkeys as leaves
    subkeys_pytree = jax.tree.unflatten(treedef, subkeys)

    return subkeys_pytree


class ContinualBackpropTrainState(TrainState):
    """
    Continual Backprop implementation in JAX.

    We store our utils in a dictionary that is structured much like network_params.
    each key tells you what network it's a part of and the index of the node.
    So a_1 corresponds to the actor, nodes corresponding to layer 1 (layer 0 is the input layer).
    This means that the output weights to layer i, node j are network_params['params']['Actor'][f'a_{i}']['kernel'][j, :]
    which means that the sum for each node should be the sum over axis=-1.
    """
    utils: struct.field(pytree_node=True)
    ages: struct.field(pytree_node=True)
    acc_num_replacements: struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        og_ts = TrainState.create(apply_fn=apply_fn, params=params, tx=tx, **kwargs)

        def is_util_leaf(tree):
            return ('kernel' in tree) and ('bias' in tree)

        def filter_input_and_bias(k, x):
            if k[-1].key[-1] != '0' and k[-1].key != 'log_std':
                # we do shape[0], since we index layer + 1
                return jnp.zeros(x['kernel'].shape[0])

        def layer_count(k, x):
            if k[-1].key[-1] != '0' and k[-1].key != 'log_std':
                # we do shape[0], since we index layer + 1
                return jnp.zeros(1)

        # we init our utilities.
        utils = jax.tree_util.tree_map_with_path(filter_input_and_bias, og_ts.params['params'], is_leaf=is_util_leaf)
        ages = jax.tree_util.tree_map_with_path(filter_input_and_bias, og_ts.params['params'], is_leaf=is_util_leaf)
        acc_num_replacements = jax.tree_util.tree_map_with_path(layer_count, og_ts.params['params'], is_leaf=is_util_leaf)

        return ContinualBackpropTrainState(
            step=og_ts.step,
            apply_fn=og_ts.apply_fn,
            params=og_ts.params,
            tx=og_ts.tx,
            opt_state=og_ts.opt_state,
            utils=utils,
            ages=ages,
            acc_num_replacements=acc_num_replacements,
            **kwargs,
        )

    def update_and_reinit(self,
                          rng: chex.PRNGKey,
                          activations: dict,
                          replacement_rate: float = 1e-4,
                          decay_rate: float = 0.99,
                          maturity_threshold: float = int(1e2)):
        # we take the mean over the batch dimension. when num_env == 1, this doesn't matter.
        activations = jax.tree.map(lambda x: jnp.mean(x, axis=0), activations)

        # First we update our ages
        new_ages = jax.tree.map(lambda x: x + 1, self.ages)

        def get_output_weight_mags(keys, u):
            # This should ignore all input keys.
            if u is None:
                return u

            target_param_set = self.params['params']
            for k in keys[:-1]:
                target_param_set = target_param_set[k.key]

            out_param_set = target_param_set[keys[-1].key]['kernel']

            # Hmmmm paper does a sum instead of a mean here. It should be the same since you're
            # meaning over the same number every time.
            output_weight_mags = jnp.abs(out_param_set).mean(axis=-1)
            return output_weight_mags

        output_weight_mags = jax.tree_util.tree_map_with_path(get_output_weight_mags, self.utils)

        # Calculate our new utils
        u = jax.tree.map(lambda h, w: jnp.abs(h) + w, activations, output_weight_mags)
        new_utils = jax.tree.map(lambda ut, u: decay_rate * ut + (1 - decay_rate) * u, self.utils, u)

        # THIS PART has a discrepancy between the code and the paper.
        # Paper says to replace one feature at a time
        # Now we figure out who CAN update first
        eligibility_mask = jax.tree.map(lambda age: age > maturity_threshold, new_ages)
        num_new_features = jax.tree.map(lambda x: replacement_rate * x.sum(), eligibility_mask)
        floor_num_new_features = jax.tree.map(lambda x: jnp.floor(x).astype(int), num_new_features)
        new_acc_num_replacements = jax.tree.map(lambda acc, nnf, fnnf: acc + nnf - fnnf,
                                                self.acc_num_replacements,
                                                num_new_features,
                                                floor_num_new_features)

        # mask of all the features we need to replace
        replacement_mask = jax.tree.map(num_top_mask, eligibility_mask, new_utils, floor_num_new_features)
        # new utils and ages for elements being replaced
        new_utils = jax.tree.map(lambda m, u: (1 - m) * u, replacement_mask, new_utils)
        new_ages = jax.tree.map(lambda m, a: (1 - m) * a, replacement_mask, new_ages)

        # helper fn to reinit params based on replacement_mask
        def replace_in_and_out(keys, og_params, rng,
                               rand_init_kernel: bool = True):
            final_key = keys[-1].key  # either kernel or bias or log_std, which we ignore
            if final_key == 'log_std':
                return og_params

            layer_key = keys[-2].key

            rmask = replacement_mask
            # we have 1: here b/c of 'params' key
            for k in keys[1:-2]:
                rmask = rmask[k.key]

            # gain (relu) * sqrt(3 / in_features)
            bound = jnp.sqrt(2) * jnp.sqrt(3 / og_params.shape[0])

            # we get our OUT mask here
            updated_params = og_params
            if final_key != 'bias' and rmask[layer_key] is not None:
                out_mask = rmask[layer_key]  # num_features
                out_mask = out_mask[..., None].repeat(og_params.shape[-1], axis=-1)
                # print the number of true values in out_mask
                # jax.debug.print("Number of features to replace: {num_replaced}", num_replaced=jnp.sum(out_mask))
                updated_params = (1 - out_mask) * updated_params
                # if rand_init_kernel:
                #     rng, _rng = jax.random.split(rng)
                #     random_init = jax.random.uniform(_rng, shape=out_mask.shape, minval=-bound, maxval=bound)
                #     updated_params += out_mask * random_init

            # we get our IN mask here. We update it based on the fact that final_key and og_params
            # gives the PREVIOUS layer.
            # first we get key for our in features for the next layer
            network_id, layer_num = layer_key.split('_')
            next_layer_key = f'{network_id}_{int(layer_num) + 1}'

            # This should filter out the last layer
            if next_layer_key in rmask:
                in_mask = rmask[next_layer_key]
                if final_key == 'kernel':
                    in_mask = in_mask[None, ...].repeat(og_params.shape[0], axis=0)

                    updated_params = (1 - in_mask) * updated_params
                    # if rand_init_kernel:
                    #     rng, _rng = jax.random.split(rng)
                    #     random_init = jax.random.uniform(_rng, shape=in_mask.shape, minval=-bound, maxval=bound)
                    #     updated_params = in_mask * random_init
                elif final_key == 'bias':
                    # zero out the biases to replace
                    updated_params = (1 - in_mask) * updated_params

            return updated_params

        rngs = generate_seeds_for_pytree(rng, self.params)
        param_replace_fn = partial(replace_in_and_out, rand_init_kernel=True)
        new_params = jax.tree_util.tree_map_with_path(param_replace_fn, self.params, rngs)

        # zero out optimizer states related to replaced nodes
        empty_state, op_state = self.opt_state
        grad_state, sched_state = op_state
        opt_state_replace_fn = partial(replace_in_and_out, rand_init_kernel=False)
        if isinstance(grad_state, ScaleByAdamState):
            # zero out adam state parameters according to replacement_mask
            new_count = jax.tree_util.tree_map_with_path(opt_state_replace_fn, grad_state.count, rngs)
            new_mu = jax.tree_util.tree_map_with_path(opt_state_replace_fn, grad_state.mu, rngs)
            new_nu = jax.tree_util.tree_map_with_path(opt_state_replace_fn, grad_state.nu, rngs)

            new_adam_state = ScaleByAdamState(count=new_count, mu=new_mu, nu=new_nu)
            new_opt_state = (empty_state, (new_adam_state, sched_state))
        elif isinstance(grad_state, EmptyState):
            new_grad_state = grad_state
            new_opt_state = (empty_state, (new_grad_state, sched_state))
        else:
            raise NotImplementedError

        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            utils=new_utils,
            ages=new_ages,
            acc_num_replacements=new_acc_num_replacements
        )