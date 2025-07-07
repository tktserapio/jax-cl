# built-in
import os
import argparse

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color


def get_max_over_bins(np_array, bin_size: int):
    """
    Gets the max over windows of size bin_size
    """
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.max(reshaped_array, axis=1)


def get_min_over_bins(np_array, bin_size: int):
    """
    Gets the min over windows of size of bin_size
    """
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.min(reshaped_array, axis=1)


def line_plot_with_shaded_region(average, standard_error, color, label):
    """
    Creates a line plot with shaded regions
    """
    line_plot_with_error_bars(results=average, error=standard_error, color=color, x_axis=np.arange(average.size) + 1,
                              light_color=lighten_color(color, 0.1), label=label)


def get_colors_combined():
    """
    Returns colors for combined PyTorch vs JAX comparison
    """
    return {
        "pytorch_base": "#d62728",      # Red for PyTorch
        "jax_base": "#1f77b4",          # Blue for JAX
        "pytorch_cbp": "#ff7f0e",       # Orange for PyTorch CBP
        "jax_cbp": "#2ca02c"            # Green for JAX CBP
    }


def retrieve_results_combined(pytorch_dir: str, jax_dir: str, metric: str, algorithm: str = "base_deep_learning_system"):
    """
    Loads results from both PyTorch and JAX implementations
    
    :param pytorch_dir: path to PyTorch results directory
    :param jax_dir: path to JAX results directory  
    :param metric: string corresponding to metric name
    :param algorithm: algorithm name (default: base_deep_learning_system)
    :return: dictionary with pytorch and jax results
    """
    
    if metric == "relative_accuracy_per_epoch":
        metric = "test_accuracy_per_epoch"

    results_dict = {}
    total_num_epochs = 4000
    epochs_per_task = 200
    denominator = 512 if "rank" in metric else 1.0
    
    # Load PyTorch results
    pytorch_path = os.path.join(pytorch_dir, algorithm, metric)
    if os.path.exists(pytorch_path):
        num_samples_pytorch = len(os.listdir(pytorch_path))
        pytorch_results = np.zeros((num_samples_pytorch, total_num_epochs // epochs_per_task), dtype=np.float32)
        
        for index in range(num_samples_pytorch):
            temp_result_path = os.path.join(pytorch_path, f"index-{index}.npy")
            if os.path.exists(temp_result_path):
                index_results = np.load(temp_result_path) / denominator
                
                if "accuracy" in metric:
                    index_results = get_max_over_bins(index_results, bin_size=epochs_per_task)
                elif "loss" in metric:
                    index_results = get_min_over_bins(index_results, bin_size=epochs_per_task)
                
                pytorch_results[index] = index_results
        
        results_dict["pytorch_base"] = pytorch_results
    
    # Load JAX results
    jax_path = os.path.join(jax_dir, algorithm, metric)
    if os.path.exists(jax_path):
        num_samples_jax = len(os.listdir(jax_path))
        jax_results = np.zeros((num_samples_jax, total_num_epochs // epochs_per_task), dtype=np.float32)
        
        for index in range(num_samples_jax):
            temp_result_path = os.path.join(jax_path, f"index-{index}.npy")
            if os.path.exists(temp_result_path):
                index_results = np.load(temp_result_path) / denominator
                
                if "accuracy" in metric:
                    index_results = get_max_over_bins(index_results, bin_size=epochs_per_task)
                elif "loss" in metric:
                    index_results = get_min_over_bins(index_results, bin_size=epochs_per_task)
                
                jax_results[index] = index_results
        
        results_dict["jax_base"] = jax_results
    
    return results_dict


def plot_combined_results(results_dict: dict, colors: dict, metric: str):
    """
    Makes a combined line plot comparing PyTorch vs JAX
    
    :param results_dict: dictionary of (implementation_algorithm, results) pairs
    :param colors: dictionary of (implementation_algorithm, color) pairs  
    :param metric: str corresponding to the metric being plotted
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for impl_alg, results in results_dict.items():
        results_mean = np.average(results, axis=0)
        results_std = np.zeros_like(results_mean)
        num_samples = results.shape[0]
        
        if num_samples > 1:
            results_std = np.std(results, axis=0, ddof=1) / np.sqrt(num_samples)
        
        # Create nice labels
        if "pytorch" in impl_alg:
            label = f"PyTorch ({impl_alg.split('_')[1].upper()})"
        else:
            label = f"JAX ({impl_alg.split('_')[1].upper()})"
        
        line_plot_with_shaded_region(results_mean, results_std, colors[impl_alg], label=label)
    
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_xlabel("Task Number", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"PyTorch vs JAX Comparison: {metric.replace('_', ' ').title()}", fontsize=14)
    ax.legend(fontsize=11)
    
    # Add task boundaries
    for task in range(1, 21):  # 20 tasks total
        ax.axvline(x=task, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)


def create_combined_plots(pytorch_dir: str, jax_dir: str, metric: str, algorithm: str = "base_deep_learning_system"):
    """
    Creates combined plots comparing PyTorch and JAX implementations
    """
    
    colors = get_colors_combined()
    results = retrieve_results_combined(pytorch_dir, jax_dir, metric, algorithm)
    
    if not results:
        print(f"No results found for metric: {metric}")
        print(f"PyTorch dir: {pytorch_dir}")
        print(f"JAX dir: {jax_dir}")
        return
    
    plot_combined_results(results, colors, metric)
    
    # Save the plot
    output_filename = f"combined_{metric}_pytorch_vs_jax.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs JAX CIFAR results")
    parser.add_argument("--pytorch_results_dir", action="store", type=str, 
                        default="/users/tserapio/loss-of-plasticity/lop/incremental_cifar/results/",
                        help="Path to PyTorch results directory")
    parser.add_argument("--jax_results_dir", action="store", type=str,
                        default="/users/tserapio/jax_cl/impl_nnx/results/", 
                        help="Path to JAX results directory")
    parser.add_argument("--metric", action="store", type=str, default="test_accuracy_per_epoch",
                        help="Metric to plot",
                        choices=["test_accuracy_per_epoch", "test_loss_per_epoch", 
                                "train_accuracy_per_checkpoint", "train_loss_per_checkpoint"])
    parser.add_argument("--algorithm", action="store", type=str, default="base_deep_learning_system",
                        help="Algorithm to compare")
    
    args = parser.parse_args()
    
    create_combined_plots(
        pytorch_dir=args.pytorch_results_dir,
        jax_dir=args.jax_results_dir, 
        metric=args.metric,
        algorithm=args.algorithm
    )


if __name__ == "__main__":
    main()