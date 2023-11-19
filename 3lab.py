import pandas as pd
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull


# Read the CSV file into a DataFrame
csv_file_path = 'C:/Users/Lenovo/Desktop/3lab masininis/Wholesale customers data.csv'
df = pd.read_csv(csv_file_path)

# Drop the 'Region' column
df = df.drop('Region', axis=1)
# Replace values in the 'Channel' column
df['Channel'] = df['Channel'].replace({1: 'Horeca', 2: 'Retail'})

# Convert 'Channel' to categorical data type
df['Channel'] = pd.Categorical(df['Channel'])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Extract features for dimensionality reduction
features1 = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features1]

# Normalised data for dimensionality reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features1)

# Extract features for clustering
features2 = ['Fresh', 'Grocery']
Xclust = df[features2]

# Normalised data for clustering
Xclust_scaled = scaler.fit_transform(Xclust)
Xclust_scaled = pd.DataFrame(Xclust_scaled, columns=features2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Create UMAP visualization with specified parameters
reducer = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0, random_state=42)
embedding = reducer.fit_transform(X_scaled)
X_scaled['Channel'] = df['Channel']

# Create a custom colormap with red and blue
cmap_custom = ListedColormap(['orange', 'blue'])

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the UMAP results with the custom colormap
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=X_scaled['Channel'].cat.codes, alpha=0.5, cmap=cmap_custom)

# Create legend handles for each category
legend_handles = [Line2D([0], [0], marker='o', color='w', label=label,
                        markerfacecolor=color, markersize=10) for label, color in zip(X_scaled['Channel'].cat.categories, cmap_custom.colors)]

# Add legend
ax.legend(handles=legend_handles, title='Channel')

ax.set_title(f'UMAP Visualization (n_neighbors=30, min_dist=0.1, spread=3.0)')
ax.set_xlabel('UMAP Dimension 1')
ax.set_ylabel('UMAP Dimension 2')

# Display the plot
plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # Elbow Method
# # Fit k-means clustering models with different values of k
# inertia = []
# for k in range(1, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X[['Fresh', 'Grocery']])
#     inertia.append(kmeans.inertia_)
# # Plot the elbow curve
# plt.plot(range(1, 11), inertia, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Sum of Squared Distances')
# plt.title('Elbow Method')
# plt.show()


# # Mean Silhouette Method
# silhouette_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X[['Fresh', 'Grocery']])
#     silhouette_avg = silhouette_score(X[['Fresh', 'Grocery']], cluster_labels)
#     silhouette_scores.append(silhouette_avg)
# # Plot silhouette scores
# plt.plot(range(2, 11), silhouette_scores, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.title('Mean Silhouette Method')
# plt.show()


# # Davies-Bouldin Index
# davies_bouldin_scores = []
# for k in range(2, 11):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(X[['Fresh', 'Grocery']])
#     davies_bouldin = davies_bouldin_score(X[['Fresh', 'Grocery']], cluster_labels)
#     davies_bouldin_scores.append(davies_bouldin)
# # Plot Davies-Bouldin scores
# plt.plot(range(2, 11), davies_bouldin_scores, marker='o')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Davies-Bouldin Index')
# plt.title('Davies-Bouldin Index Method')
# plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Fit the nearest neighbors model
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(Xclust_scaled)
distances, indices = nbrs.kneighbors(Xclust_scaled)

# Sort the distances and plot the knee
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.xlabel('Data Point Index')
plt.ylabel('Distance to Nearest Neighbor')
plt.title('Knee Method for Optimal eps')
plt.show()

# Manually choose eps based on the knee point
# eps =  0.25

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.25, min_samples=3)  # You can adjust min_samples based on your data
clusters = dbscan.fit_predict(Xclust_scaled)

# Add the cluster labels to the scaled DataFrame
Xclust_scaled['Cluster'] = clusters

# Scatter plot
plt.figure(figsize=(10, 6))

# Create a dictionary to map cluster labels to colors
cluster_colors = {
    -1: '#800080',  # Purple
    0: '#008080',   # Teal
    1: '#008000',   # Green
    2: '#FFD700',   # Gold
    3: '#FF6347',   # Tomato
    4: '#9932CC',   # DarkOrchid
    5: '#87CEEB',   # SkyBlue
    6: '#FFA07A',   # LightSalmon
    7: '#20B2AA',   # LightSeaGreen
    8: '#9370DB',   # MediumPurple
    9: '#ADFF2F',   # GreenYellow
}

# Plotting scatter points
sns.scatterplot(x='Fresh', y='Grocery', hue='Cluster', data=Xclust_scaled, palette=cluster_colors, s=60)
plt.title('DBSCAN Clustering')
plt.xlabel('Fresh')
plt.ylabel('Grocery')
plt.legend(fontsize='large')

# Plotting convex hulls around clusters
for cluster_label in np.unique(clusters):
    if cluster_label != -1:
        cluster_points = Xclust_scaled.loc[Xclust_scaled['Cluster'] == cluster_label, ['Fresh', 'Grocery']].values
        
        if len(cluster_points) > 0:
            hull = ConvexHull(cluster_points)
            
            # Get the color of the cluster from the dictionary
            cluster_color = cluster_colors.get(cluster_label, 'black')
            
            # Plot filled convex hull with low opacity and darker outline
            plt.fill(np.append(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 0][0]),
                     np.append(cluster_points[hull.vertices, 1], cluster_points[hull.vertices, 1][0]),
                     color=cluster_color, alpha=0.3, edgecolor=cluster_color, linewidth=2)

plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Fit the nearest neighbors model
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(embedding)
distances, indices = nbrs.kneighbors(embedding)

# Sort the distances and plot the knee
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.xlabel('Data Point Index')
plt.ylabel('Distance to Nearest Neighbor')
plt.title('Knee Method for Optimal eps')
plt.show()

# Manually choose eps based on the knee point
# eps =  0.6

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Perform DBSCAN clustering on the UMAP-reduced data
dbscan = DBSCAN(eps=0.6, min_samples=10)
clusters = dbscan.fit_predict(embedding)

# Add cluster labels to the UMAP-reduced DataFrame
umap_df = pd.DataFrame(embedding, columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
umap_df['Cluster'] = clusters

# Scatter plot with different colors for each cluster
plt.figure(figsize=(10, 6))

# Plotting scatter points with different colors for each cluster
sns.scatterplot(x='UMAP Dimension 1', y='UMAP Dimension 2', hue='Cluster', palette=cluster_colors, data=umap_df, s=60)

# Plotting convex hulls around clusters
for cluster_label in np.unique(clusters):
    if cluster_label != -1:
        cluster_points = umap_df.loc[umap_df['Cluster'] == cluster_label, ['UMAP Dimension 1', 'UMAP Dimension 2']].values
        
        if len(cluster_points) > 0:
            hull = ConvexHull(cluster_points)
            plt.fill(np.append(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 0][0]),
                     np.append(cluster_points[hull.vertices, 1], cluster_points[hull.vertices, 1][0]),
                     color=cluster_colors[cluster_label], alpha=0.3, edgecolor=cluster_colors[cluster_label], linewidth=2)

# Plot titles and labels
plt.title('UMAP + DBSCAN Clustering')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Cluster', fontsize='large')

# Show the plot
plt.show()