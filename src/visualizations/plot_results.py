import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px

def plot_age(df):
    attribute = 'Age'

    # Create a histogram using Seaborn
    plt.figure(figsize=(8, 6))
    sns.histplot(df[attribute], bins=20, kde=True, color='blue')
    plt.title(f'Distribution of {attribute}')
    plt.xlabel(attribute)  # X-axis label
    plt.ylabel('Frequency')  # Y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines
    plt.show()

def plot_gender(df):
    attribute = 'gender'
    df[attribute] = df[attribute].str.strip().str.lower().replace({'unknown': None, 'other': None})
    # Create a histogram using Seaborn
    sns.countplot(x='gender', data=df, palette='Set2')
    plt.title(f'Distribution of {attribute}')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.show()

def pairplot_k_means(df):
    pca_columns = [col for col in df.columns if col.startswith("PCA_")]
    sns.pairplot(df, vars=pca_columns, hue='cluster', palette='viridis')
    plt.suptitle("Pairplot of PCA Components by Cluster", y=1.02)
    plt.show()
    plt.savefig(r"F:\University\Uni Stuff (semester 11)\Thesis\code\reports\figures\Pairplot of PCA Components by Cluster.png")
    pass

def hotspots_per_cluster_plot(df):
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=df, 
        x='cluster', 
        y='crime_count', 
        hue='Crime_Location_raw', 
        dodge=False, 
        palette="Paired"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Crime Count")
    plt.title("Crime Hotspots Per Cluster")
    plt.xticks(rotation=45)  # Rotate x labels for better readability
    plt.legend(title="Crime Location", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.savefig(r"F:\University\Uni Stuff (semester 11)\Thesis\code\reports\figures\hotspots_per_cluster_plot.png")
    plt.show()

def visualize_clusters(df_final, save_path=None):
    # Visualize clusters in 3D using t-SNE with an option to save the plot.
    
    # Parameters:
    # df_final (pd.DataFrame): DataFrame containing features and 'cluster' column
    # save_path (str): Optional path to save the plot (supports .html, .png, .jpg, .jpeg)
    # Perform t-SNE dimensionality reduction to 3D
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_results = tsne.fit_transform(df_final.drop(columns=['cluster']))
    
    # Add t-SNE coordinates to DataFrame
    df_final[['tsne-1', 'tsne-2', 'tsne-3']] = tsne_results
    
    # Create 3D scatter plot
    fig = px.scatter_3d(df_final,
                        x='tsne-1',
                        y='tsne-2',
                        z='tsne-3',
                        color=df_final['cluster'].astype(str),
                        title="3D t-SNE Cluster Visualization",
                        labels={'color': 'Cluster'},
                        hover_data=df_final.columns,
                        height=800,)
    
    # Customize layout
    fig.update_layout(scene=dict(
                        xaxis_title='t-SNE 1',
                        yaxis_title='t-SNE 2',
                        zaxis_title='t-SNE 3'),
                      margin=dict(l=0, r=0, b=0, t=30))
    
    # Save or show the plot
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        elif save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            fig.write_image(save_path)
        else:
            raise ValueError("Unsupported file format. Use .html, .png, .jpg, or .jpeg")
        print(f"Plot saved to {save_path}")
    else:
        fig.show()