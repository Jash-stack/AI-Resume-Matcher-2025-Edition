import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from sentence_transformers import SentenceTransformer

# --- Cluster Jobs ---
def cluster_jobs(jobs_df, num_clusters=3):
    if jobs_df.empty or len(jobs_df) < num_clusters:
        return jobs_df, None

    if jobs_df['Description'].isnull().all():
        raise ValueError("All job descriptions are empty. Cannot perform clustering.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(jobs_df['Description'].fillna(""), show_progress_bar=False)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    jobs_df['Cluster'] = labels

    return jobs_df, visualize_clusters(jobs_df, X, labels)

# --- Plotly Visualization ---
def visualize_clusters(jobs_df, features, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    df_plot = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    df_plot['Cluster'] = labels
    df_plot['Job Title'] = jobs_df['Job Title']
    df_plot['Company'] = jobs_df['Company']

    fig = px.scatter(
        df_plot,
        x='PC1', y='PC2',
        color=df_plot['Cluster'].astype(str),
        hover_data=['Job Title', 'Company'],
        title="🔍 Job Clusters (Semantic PCA View)"
    )
    fig.update_layout(height=500)
    return fig