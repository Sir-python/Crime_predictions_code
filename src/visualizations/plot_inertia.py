import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_inertia(df):
    inertia = []
    k_range = range(1, 11)  # Test K from 1 to 10

    # Running K-Means for Different Cluster Counts
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    # Inertia plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o', linestyle='--', color='b')

    # Set axis labels and title with increased font sizes
    plt.xlabel("Number of Clusters (K)", fontsize=14)
    plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=14)
    plt.title("Elbow Method for Optimal K", fontsize=16)

    # Set x-axis ticks to show each value in K_range
    plt.xticks(list(k_range), fontsize=12)
    plt.yticks(fontsize=12)

    # Enable a grid for both major and minor ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Optionally, you can add a minor tick locator if needed:
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(r"F:\University\Uni Stuff (semester 11)\Thesis\code\reports\figures\elbow_method_plot.png")
    plt.show()
