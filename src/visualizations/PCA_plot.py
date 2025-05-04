import matplotlib.pyplot as plt

def plot_pca_variance(explained_variance):
    # Plot cumulative variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance Explained')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.legend()
    plt.grid()
    plt.show()
