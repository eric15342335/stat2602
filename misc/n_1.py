import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")

def simulate_variance_estimation(population_mean, population_variance, sample_sizes, num_samples):
    """
    Simulate variance estimation for different sample sizes.

    Parameters:
    - population_mean: Mean of the population distribution.
    - population_variance: Variance of the population distribution.
    - sample_sizes: List of sample sizes to simulate.
    - num_samples: Number of samples to draw for each sample size.

    Returns:
    - results: Dictionary containing calculated variances.
    """
    results = {
        'sample_size': [],
        'variance_div_n': [],
        'variance_div_n_minus_1': []
    }

    for n in sample_sizes:
        var_n = []
        var_n_minus_1 = []
        for _ in range(num_samples):
            # Draw a random sample from the normal distribution
            sample = np.random.normal(loc=population_mean, scale=np.sqrt(population_variance), size=n)
            # Calculate sample mean
            sample_mean = np.mean(sample)
            # Calculate deviations from the mean
            deviations = sample - sample_mean
            # Calculate variance using denominator n (biased estimator)
            variance_n = np.sum(deviations ** 2) / n
            # Calculate variance using denominator n - 1 (unbiased estimator)
            variance_n_minus_1 = np.sum(deviations ** 2) / (n - 1)
            # Store the results
            var_n.append(variance_n)
            var_n_minus_1.append(variance_n_minus_1)
        # Append results for the current sample size
        results['sample_size'].extend([n] * num_samples)
        results['variance_div_n'].extend(var_n)
        results['variance_div_n_minus_1'].extend(var_n_minus_1)

    return results

def plot_variance_estimates(results, population_variance):
    """
    Plot the distributions of variance estimates.

    Parameters:
    - results: Dictionary containing calculated variances.
    - population_variance: The true variance of the population.
    """
    sample_sizes = sorted(set(results['sample_size']))

    for n in sample_sizes:
        # Extract variances for current sample size
        indices = [i for i, size in enumerate(results['sample_size']) if size == n]
        var_n = [results['variance_div_n'][i] for i in indices]
        var_n_minus_1 = [results['variance_div_n_minus_1'][i] for i in indices]

        # Create subplots
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms of variance estimates
        sns.histplot(var_n, color='red', label='Dividing by n (Biased)', kde=True, stat="density", ax=ax)
        sns.histplot(var_n_minus_1, color='blue', label='Dividing by n - 1 (Unbiased)', kde=True, stat="density", ax=ax)

        # Plot vertical line for true variance
        ax.axvline(population_variance, color='green', linestyle='--', label='True Variance')

        # Set plot titles and labels
        ax.set_title(f'Variance Estimates for Sample Size n = {n}')
        ax.set_xlabel('Variance Estimate')
        ax.set_ylabel('Density')
        ax.legend()

        plt.show()

def main():
    # Parameters for the population distribution
    population_mean = 0
    population_variance = 1  # Standard normal distribution

    # Sample sizes to simulate
    sample_sizes = [5, 10, 30, 100]

    # Number of samples to draw for each sample size
    num_samples = 10000

    # Simulate variance estimation
    results = simulate_variance_estimation(population_mean, population_variance, sample_sizes, num_samples)

    # Plot the variance estimates
    plot_variance_estimates(results, population_variance)

if __name__ == "__main__":
    main()
