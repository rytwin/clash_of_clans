#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 16:24:31 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# read in csv files
df = pd.read_csv('data/unit_info.csv')
df = df[df['type'] == 'troops']

### just update the 3 variables below. then you can rerun the whole script and get
### the plots and metrics to evaluate the clustering for min-max and z-score scaling

# select variables for use in clustering
cluster_df = df[['unit', 'support', 'defenses', 'housing', 'dps_max', 'pot_dps_max', 'hp_max']]
k = 5 # number of clusters for actual cluster assignments (viewed in cluster_df dataframe)
k_max = 10 # max number of clusters for silhouette score, elbow, and silhouette shape plots



#################################

# create standardized variables with min/max scaling and z-score scaling
cluster_df.set_index('unit', inplace=True)
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
std_scaled_arr = std_scaler.fit_transform(cluster_df)
mm_scaled_arr = mm_scaler.fit_transform(cluster_df)

# create kMeans clustering predictions for both
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_df['std_Cluster'] = kmeans.fit_predict(std_scaled_arr)
cluster_df['mm_Cluster'] = kmeans.fit_predict(mm_scaled_arr)

# calculate silhouette score and create silhouette score graphs for both
silhouette_scores_std = []
for k in range(2, k_max):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_df)
    silhouette_scores_std.append(silhouette_score(std_scaled_arr, kmeans.labels_))

plt.plot(range(2, k_max), silhouette_scores_std, marker='o', color='blue')
plt.title('Silhouette Score (standard scaler)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

silhouette_scores_mm = []
for k in range(2, k_max):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_df)
    silhouette_scores_mm.append(silhouette_score(mm_scaled_arr, kmeans.labels_))

plt.plot(range(2, k_max), silhouette_scores_mm, marker='o', color='red')
plt.title('Silhouette Score (min-max scaler)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

# calculate inertia and plot elbow method for both
inertias_std = []
for k in range(1, k_max):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(std_scaled_arr)
    inertias_std.append(kmeans.inertia_)

plt.plot(range(1, k_max), inertias_std, marker='o', color='blue')
plt.title('Elbow Method (standard scaler)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

inertias_mm = []
for k in range(1, k_max):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mm_scaled_arr)
    inertias_mm.append(kmeans.inertia_)

plt.plot(range(1, k_max), inertias_mm, marker='o', color='red')
plt.title('Elbow Method (min-max scaler)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# create silhouette charts for each type of clusters
k_range = np.arange(2, k_max)
for n_clusters in k_range:
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
    
    # for standard scaled variables
    ax1.set_ylim([0, len(cluster_df) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(std_scaled_arr)

    silhouette_avg = silhouette_score(std_scaled_arr, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sample_silhouette_values = silhouette_samples(std_scaled_arr, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("Standard scaler")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    
    # for min-max scaled variables
    ax2.set_ylim([0, len(cluster_df) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(mm_scaled_arr)

    silhouette_avg = silhouette_score(mm_scaled_arr, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sample_silhouette_values = silhouette_samples(mm_scaled_arr, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):        
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax2.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax2.set_title("Min-max scaler")
    ax2.set_xlabel("Silhouette coefficient values")
    ax2.set_ylabel("Cluster label")
    ax2.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax2.set_yticks([])
    
    fig.tight_layout()
    plt.suptitle(
        "n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

