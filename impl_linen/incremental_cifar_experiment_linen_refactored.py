import argparse
import json
import os
import time
from functools import partialmethod

# third‑party
import numpy as np
from tqdm import tqdm

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

# mlproj‑manager helpers
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import (
    access_dict,
    get_random_seeds,
    turn_off_debugging_processes,
)

# Local imports
from cifar_data_handler import CIFARDataHandler
from incremental_trainer import IncrementalTrainer, TrainState
from torchvision_modified_resnet_linen import build_resnet18, kaiming_init_resnet_module


class IncrementalCIFARExperimentJAX(Experiment):
    """
    Simplified Flax/JAX Incremental CIFAR-100 Experiment Runner.
    Handles configuration, setup, and orchestration. All training logic is in IncrementalTrainer.
    """

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose: bool = True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # Debug control
        debug = access_dict(exp_params, "debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # Device setup
        self.device = self._setup_device()
        
        # Progress bar setup
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)

        # Reproducibility
        random_seeds = get_random_seeds()
        self.random_seed = int(random_seeds[run_index])
        np.random.seed(self.random_seed)

        # Extract and validate experiment parameters
        self._extract_experiment_params(exp_params)
        
        # Initialize data handler
        self.data_handler = CIFARDataHandler(
            data_path=self.data_path,
            num_workers=self.num_workers,
            batch_sizes=self.batch_sizes,
            device=None  # JAX doesn't need explicit device placement for data
        )
        
        # Build and initialize model
        self.model = self._build_model()
        self.state = self._initialize_model_state()
        
        # Initialize trainer with all training logic
        self.trainer = self._initialize_trainer()

    def _setup_device(self):
        """Setup JAX device (GPU or CPU)."""
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                device = gpu_devices[0]
                print(f"✅ JAX is using GPU: {device}")
                print(f"   Available GPU devices: {len(gpu_devices)}")
                for i, gpu_device in enumerate(gpu_devices):
                    print(f"   GPU {i}: {gpu_device}")
            else:
                device = jax.devices("cpu")[0]
                print(f"⚠️  JAX is using CPU: {device}")
                print("   No GPU devices found")
        except RuntimeError as e:
            device = jax.devices("cpu")[0]
            print(f"⚠️  JAX is using CPU: {device}")
            print(f"   GPU detection failed: {e}")
        return device

    def _extract_experiment_params(self, exp_params: dict):
        """Extract and validate all experiment parameters."""
        # Data parameters
        self.data_path: str = exp_params["data_path"]
        self.num_workers: int = access_dict(exp_params, "num_workers", default=1, val_type=int)
        
        # Training hyperparameters
        self.stepsize: float = float(exp_params["stepsize"])
        self.weight_decay: float = float(exp_params["weight_decay"])
        self.momentum: float = float(exp_params["momentum"])

        # Reset flags
        self.reset_head: bool = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network: bool = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print("Warning: Resetting whole network supersedes resetting head.")
        self.early_stopping: bool = access_dict(exp_params, "early_stopping", default=False, val_type=bool)

        # CBP parameters
        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert (not self.use_cbp) or (self.replacement_rate > 0.0), "Replacement rate should be greater than 0."
        self.utility_function = access_dict(exp_params, "utility_function", default="weight", val_type=str,
                                            choices=["weight", "contribution"])
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert (not self.use_cbp) or (self.maturity_threshold > 0), "Maturity threshold should be greater than 0."
       
        # Noise injection
        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)

        # Experiment constants
        self.num_epochs = 4000
        self.num_classes = 100
        self.batch_sizes = {"train": 90, "validation": 50, "test": 100}
        self.image_dims = (32, 32, 3)
        self.class_increase_frequency = 200

    def _build_model(self):
        """Build and initialize the ResNet model."""
        model = build_resnet18(num_classes=self.num_classes, norm_layer=nn.BatchNorm)
        return model

    def _initialize_model_state(self):
        """Initialize model state with parameters and optimizer."""
        # Initialize model parameters
        key = jax.random.PRNGKey(self.random_seed)
        key, init_key = jax.random.split(key)
        
        dummy = jnp.ones((1, *self.image_dims), jnp.float32)
        variables = self.model.init(init_key, dummy, feature_list=None, train=True)
        params, batch_stats = variables["params"], variables.get("batch_stats", {})
        
        # Apply Kaiming initialization
        params = kaiming_init_resnet_module(params)

        # Initialize optimizer
        optim = optax.sgd(learning_rate=self.stepsize, momentum=self.momentum)
        
        # Create training state
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optim,
            batch_stats=batch_stats
        )
        
        return state

    def _initialize_trainer(self):
        """Initialize the incremental trainer with all training logic."""
        trainer = IncrementalTrainer(
            model=self.model,
            stepsize=self.stepsize,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            noise_std=self.noise_std,
            use_cbp=self.use_cbp,
            replacement_rate=self.replacement_rate,
            utility_function=self.utility_function,
            maturity_threshold=self.maturity_threshold,
            reset_head=self.reset_head,
            reset_network=self.reset_network,
            early_stopping=self.early_stopping,
            results_dir=self.results_dir,
            run_index=self.run_index,
            verbose_print_fn=self._print,
            num_epochs=self.num_epochs,
            num_classes=self.num_classes,
            class_increase_frequency=self.class_increase_frequency,
            batch_sizes=self.batch_sizes,
            image_dims=self.image_dims,
            random_seed=self.random_seed,
            device=self.device
        )
        
        # Set the trainer's state
        trainer.state = self.state
        
        return trainer

    def run(self):
        """Main experiment runner - orchestrates data loading and training."""
        # Load data using the data handler
        train_ds, train_dl = self.data_handler.get_data(train=True, validation=False)
        val_ds, val_dl = self.data_handler.get_data(train=True, validation=True)
        test_ds, test_dl = self.data_handler.get_data(train=False)
        
        # Run training via the trainer
        self.trainer.train(train_dl, test_dl, val_dl, train_ds, test_ds, val_ds)
        
        # Store results using parent class method
        # Transfer results from trainer to experiment for storage
        self.results_dict = self.trainer.results_dict
        self.store_results()


def main():
    """Main entry point wrapper."""
    parser = argparse.ArgumentParser(description="Incremental CIFAR‑100 Experiment (Flax)")
    parser.add_argument("--config", type=str, default="./incremental_cifar/cfg/base_deep_learning_system.json")
    parser.add_argument("--experiment-index", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    base = os.path.dirname(os.path.abspath(__file__))
    
    # Fix data_path handling - check for empty string as well as missing key
    if "data_path" not in cfg or not cfg["data_path"]:
        cfg["data_path"] = os.path.join(base, "data")
    if "results_dir" not in cfg or not cfg["results_dir"]:
        cfg["results_dir"] = os.path.join(base, "results")
    if "experiment_name" not in cfg or not cfg["experiment_name"]:
        cfg["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]
    
    # Debug: Print the paths being used
    print(f"Using data_path: '{cfg['data_path']}'")
    print(f"Using results_dir: '{cfg['results_dir']}'")
    print(f"Using experiment_name: '{cfg['experiment_name']}'")
    
    # Make sure data directory exists
    os.makedirs(cfg["data_path"], exist_ok=True)

    exp = IncrementalCIFARExperimentJAX(
        cfg,
        results_dir=os.path.join(cfg["results_dir"], cfg["experiment_name"]),
        run_index=args.experiment_index,
        verbose=args.verbose,
    )

    start = time.perf_counter()
    exp.run()
    end = time.perf_counter()
    print(f"Total runtime: {(end - start) / 60:.2f} min")


if __name__ == "__main__":
    main()
