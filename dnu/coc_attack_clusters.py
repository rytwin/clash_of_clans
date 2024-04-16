#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 23:26:14 2024

@author: ryandrost
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

df = pd.read_csv('attack_data_for_model.csv')


# create df for clustering to identify attack types
cluster_df = df[['attack_id', 'aq', 'bk', 'gw', 'rc', 'myak', 'sfox', 'frosty',
    'diggy', 'pliz', 'eowl', 'phoenix', 'lassi', 'unicorn',
    'gg', 'rvial', 'barbpup', 'eqboots', 'vstache', 'archpup', 'healpup', 'ivial', 'arrow',
    'htome', 'etome', 'rgem', 'lgem', 'sshield', 'roygem',
    'barb_pct', 'ram_pct', 'cookie_pct', 'arch_pct',
    'giant_pct', 'gob_pct', 'wb_pct', 'wiz_pct', 'balloon_pct',
    'healer_pct', 'drag_pct', 'pekka_pct', 'babydrag_pct',
    'miner_pct', 'edrag_pct', 'yeti_pct', 'drider_pct',
    'etitan_pct', 'root_pct', 'minion_pct', 'hog_pct',
    'valk_pct', 'golem_pct', 'witch_pct', 'lh_pct',
    'bowl_pct', 'ig_pct', 'hh_pct', 'aw_pct',
    'supbarb_pct', 'suparch_pct', 'sneakgob_pct', 'supwb_pct',
    'supgiant_pct', 'rockloon_pct', 'supwiz_pct', 'supdrag_pct',
    'infdrag_pct', 'supminion_pct', 'supvalk_pct', 'supwitch_pct',
    'ih_pct', 'supbowl_pct', 'supminer_pct', 'suphog_pct',
    'light_pct', 'bof_pct', 'heal_pct', 'rage_pct',
    'jump_pct', 'freeze_pct', 'clone_pct', 'invis_pct',
    'recall_pct', 'poison_pct', 'eq_pct', 'haste_pct',
    'skel_pct', 'bat_pct', 'cc_barb_num', 'cc_ram_num', 'cc_cookie_num', 'cc_arch_num', 
    'cc_giant_num', 'cc_gob_num', 'cc_wb_num', 'cc_wiz_num', 'cc_balloon_num', 
    'cc_healer_num', 'cc_drag_num', 'cc_pekka_num', 'cc_babydrag_num',
    'cc_miner_num', 'cc_edrag_num', 'cc_yeti_num', 'cc_drider_num',
    'cc_etitan_num', 'cc_root_num', 'cc_minion_num', 'cc_hog_num',
    'cc_valk_num', 'cc_golem_num', 'cc_witch_num', 'cc_lh_num',
    'cc_bowl_num', 'cc_ig_num', 'cc_hh_num', 'cc_aw_num',
    'cc_supbarb_num', 'cc_suparch_num', 'cc_sneakgob_num', 'cc_supwb_num',
    'cc_supgiant_num', 'cc_rockloon_num', 'cc_supwiz_num', 'cc_supdrag_num',
    'cc_infdrag_num', 'cc_supminion_num', 'cc_supvalk_num', 'cc_supwitch_num',
    'cc_ih_num', 'cc_supbowl_num', 'cc_supminer_num', 'cc_suphog_num',
    'ww_num', 'bb_num', 'ss_num', 'sb_num', 'll_num', 'ff_num', 'bd_num']]

cluster_df.set_index('attack_id', inplace=True)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_df['Cluster'] = kmeans.fit_predict(cluster_df)

silhouette_scores = []
for k in range(2, 31):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_df)
    silhouette_scores.append(silhouette_score(cluster_df, kmeans.labels_))

plt.plot(range(2, 31), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()


inertias = []
for k in range(1, 31):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_df)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 31), inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

range_n_clusters = [6]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, ax = plt.subplots(1, 1)

    # The 1st subplot is the silhouette plot

    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(cluster_df) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(cluster_df)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(cluster_df, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(cluster_df, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

