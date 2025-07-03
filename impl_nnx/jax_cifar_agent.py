class EffectiveRankAgent:
    def __init__(self, network: DeepFFNN):
        self.network = network
        self.loss = jax.jit(self.loss)
        self.effective_rank_loss = jax.jit(self.effective_rank_loss)
    
    def predict(self, params, x):
        output, features = self.network.apply(params, x)
        return output, features
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, x):
        output, features = self.network.apply(params, x)
        erank_losses = [self.effective_rank(f) for f in features]
        loss_erank = - jnp.stack(erank_losses).mean()
        return loss_erank

    def loss(self, params, x, y):
        output, features = self.network.apply(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y))
        return loss

def make_train(args: PermutedMnistHyperparams, rng: chex.PRNGKey):
    network = build_resnet18(
    )
    # DeepFFNN(
    #     num_features=args.num_features,
    #     num_outputs=10,  # MNIST has 10 classes
    #     num_hidden_layers=args.num_hidden_layers,
    #     act_type=args.activation
    # )
    num_tasks = args.num_tasks
    images_per_class = 450
    classes_per_task = 10
    input_size = 784
    examples_per_task = images_per_class * classes_per_task
    
    def train(lr, er_lr, rng):
        agent = EffectiveRankAgent(network)
        
        # load data
        with open('data/mnist_', 'rb') as f:
            x_all, y_all, _, _ = np.load(f, allow_pickle=True)
        x_all = jnp.array(x_all)
        y_all = jnp.array(y_all)
        
        # init network
        network_params = network.init(rng, x_all[:1])

        if args.no_anneal_lr:
            if args.optimizer == 'adam':
                tx = optax.chain(
                    optax.adamw(learning_rate=lr, weight_decay=args.weight_decay)
                )
            else:
                tx = optax.chain(
                    optax.add_decayed_weights(args.weight_decay),
                    optax.sgd(learning_rate=lr)
                )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        assert (examples_per_task // args.mini_batch_size) % args.er_batch == 0, "ER batch size must divide examples per task"
        def update_task(runner_state, task):
            def update_erbatch(runner_state, batch_idx):
                def update_accuracy(runner_state, mini_batch_idx):
                    x, y, train_state, rng = runner_state
                    minibatch_x = jax.lax.dynamic_slice_in_dim(x, mini_batch_idx, args.mini_batch_size, axis=0)
                    minibatch_y = jax.lax.dynamic_slice_in_dim(y, mini_batch_idx, args.mini_batch_size, axis=0)
                    loss = agent.loss(train_state.params, minibatch_x, minibatch_y)

                    logits, _ = agent.predict(train_state.params, minibatch_x)
                    pred_labels = jnp.argmax(logits, axis=-1)
                    accuracy = jnp.mean(pred_labels == minibatch_y)

                    grads = jax.grad(agent.loss)(train_state.params, minibatch_x, minibatch_y)
                    train_state = train_state.apply_gradients(grads=grads)
                    return (x, y, train_state, rng), (loss, accuracy)
                        
                x, y, train_state, rng = runner_state
                batch_x = jax.lax.dynamic_slice_in_dim(x, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                batch_y = jax.lax.dynamic_slice_in_dim(y, batch_idx, args.mini_batch_size * args.er_batch, axis=0)
                accuracy_runner_state = (batch_x, batch_y, train_state, rng)
                accuracy_runner_state, (loss, accuracy) = jax.lax.scan(update_accuracy, accuracy_runner_state, jnp.arange(0, args.er_batch * args.mini_batch_size, args.mini_batch_size), args.er_batch)
                train_state = accuracy_runner_state[2]
                # runner_state = (x, y, train_state, rng)
                # return runner_state, (loss, accuracy)

                def update_erank(runner_state, _):
                    x, train_state, rng = runner_state
                    er_loss = agent.effective_rank_loss(train_state.params, x)
                    grads = jax.grad(agent.effective_rank_loss)(train_state.params, x)
                    train_state = train_state.apply_gradients(grads=grads)
                    return (x, train_state, rng), er_loss

                if args.agent in ['er', 'l2_er']:
                    er_runner_state = (batch_x, train_state, rng)
                    er_runner_state, er_loss = jax.lax.scan(update_erank, er_runner_state, None, args.er_step)
                    train_state = er_runner_state[1]
                
                runner_state = (x, y, train_state, rng)
                return runner_state, (loss, accuracy)

            x, y, train_state, train_previous, rng = runner_state
            # x_all, y_all, train_state, rng = runner_state
            # rng, _rng = jax.random.split(rng)
            # pixel_permutation = jax.random.permutation(rng, input_size)
            # x_all = x_all[:, pixel_permutation]
            # # Shuffle the data for the current task
            # rng, _rng = jax.random.split(rng)
            # data_permutation = jax.random.permutation(rng, examples_per_task)
            # x, y = x_all[data_permutation], y_all[data_permutation]
            update_erbatch_runner_state = (x, y, train_state, rng)
            update_erbatch_runner_state, (loss, accuracy) = jax.lax.scan(update_erbatch, update_erbatch_runner_state, 
                                        jnp.arange(0, examples_per_task, args.mini_batch_size * args.er_batch), 
                                        examples_per_task // (args.mini_batch_size * args.er_batch))
            accuracy = jnp.mean(accuracy)
            train_state = update_erbatch_runner_state[2]
            runner_state = (x_all, y_all, train_state, rng)

            # Evaluate the model on the current task
            x_eval, y_eval = x[:args.eval_size], y[:args.eval_size]
            output, features = agent.predict(train_state.params, x_eval)
            rank, effective_rank, approx_rank, approx_rank_abs, dead_neurons = summarize_all_layers(features)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_eval = jnp.mean(pred_labels == y_eval)

            # Evaluate the model on the previous train set
            x_pretrain, y_pretrain = train_previous
            output, features = agent.predict(train_state.params, x_pretrain)
            pred_labels = jnp.argmax(output, axis=-1)
            accuracy_pre = jnp.mean(pred_labels == y_pretrain)

            if args.debug:
                jax.debug.print("Task {t}: Train Accuracy {acc}, Eval Accuracy = {acc_eval}, Pretrain Accuracy = {acc_pretrain}", t=task, acc=accuracy, acc_eval=accuracy_eval, acc_pretrain=accuracy_pre)
                jax.debug.print(
                    "Rank: {r}, EffRank: {er}, ApproxRank: {ar}, DeadNeurons: {dn}",
                    r=rank, er=effective_rank, ar=approx_rank, dn=dead_neurons
                )
                
            res_info = {
                'loss': loss,
                'accuracy': accuracy,
                'rank': rank,
                'effective_rank': effective_rank,
                'approx_rank': approx_rank,
                'dead_neurons': dead_neurons,
                'accuracy_eval': accuracy_eval,
                'accuracy_pre': accuracy_pre
            }
                
            return runner_state, res_info

        loss_list, acc_list, rank_list, eff_rank_list, approx_rank_list, dead_neurons_list = [], [], [], [], [], []
        update_task = jax.jit(update_task)
        for task in range(num_tasks):
            eval_size = args.eval_size
            train_size = examples_per_task - eval_size
            # Record the previous train set
            train_previous = (x_all[train_size:], y_all[train_size:])
            # permuted dataset
            rng, _rng = jax.random.split(rng)
            pixel_permutation = jax.random.permutation(rng, input_size)
            x_all = x_all[:, pixel_permutation]
            # Shuffle the data for the current task
            rng, _rng = jax.random.split(rng)
            data_permutation = jax.random.permutation(rng, examples_per_task)
            x_shuffled, y_shuffled = x_all[data_permutation], y_all[data_permutation]

            # Split into train and eval sets
            x_train, y_train = x_shuffled[:train_size], y_shuffled[:train_size]
            x_eval, y_eval = x_shuffled[train_size:], y_shuffled[train_size:]

            #compute hessian at the start of the task
            if args.compute_hessian and task % args.compute_hessian_interval == 0:
                # Hessian computation on test set
                x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

                # Hessian computation on train set
                x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=True)



            runner_state = (
                x_train,
                y_train,
                train_state,
                train_previous, 
                rng)
            runner_state, res_info = update_task(runner_state, task)
            x_train, y_train, train_state, rng = runner_state
            rank_list.append(res_info['rank'])
            eff_rank_list.append(res_info['effective_rank'])
            approx_rank_list.append(res_info['approx_rank'])
            dead_neurons_list.append(res_info['dead_neurons'])


            #compute hessian at the end of the task
            if args.compute_hessian and task % args.compute_hessian_interval == 0:
                # TODO: Compute the Hessian
                x_hessian, y_hessian = x_eval[:args.compute_hessian_size], y_eval[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_test, grids_test = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

                # Hessian computation on train set
                x_hessian, y_hessian = x_train[:args.compute_hessian_size], y_train[:args.compute_hessian_size]
                hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, train_state.params, (x_hessian, y_hessian))
                hvp_cl = lambda v: hvp_fn(train_state.params, v)
                rng, _rng = jax.random.split(rng)
                tridiag, lanczos_vecs = lanczos_alg(
                    hvp_cl,
                    num_params,
                    order=100,
                    rng_key=rng
                )
                density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)
                jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_test, density_test, task, args.agent, at_init=False)



        final_train_state = runner_state[2]
        ranks             = jnp.stack(rank_list)
        eff_ranks         = jnp.stack(eff_rank_list)
        approx_ranks      = jnp.stack(approx_rank_list)
        dead_neurons      = jnp.stack(dead_neurons_list)

        res_info = {
            'rank':            ranks,
            'effective_rank':  eff_ranks,
            'approx_rank':     approx_ranks,
            'dead_neurons':    dead_neurons,
            'train_state':     final_train_state
        }
        return res_info
    return train

if __name__ == "__main__":
    args = PermutedMnistHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)
    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # we need to go backwards, since JAX returns indices
    # in the order in which they're vmapped.
    for i, arg in reversed(list(enumerate(train_args))):
        dims = [None] * len(train_args)
        dims[i] = 0
        vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
        if arg == 'rng':
            swept_args.appendleft(rngs)
        else:
            assert hasattr(args, arg)
            swept_args.appendleft(getattr(args, arg))

    train_jit = vmaps_train
    t = time()
    print(*swept_args)
    out = train_jit(*swept_args)
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    final_train_state = out['train_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")