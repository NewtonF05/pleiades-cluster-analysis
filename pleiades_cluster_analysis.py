#!/usr/bin/env python
# coding: utf-8

"""
Pleiades (M45) Cluster Analysis with Unsupervised Learning
==========================================================

Author: Newton Fernihough

An end-to-end analysis of Gaia astrometric data toward the Pleiades open
cluster (M45). Stars are filtered on parallax quality and proper-motion
bounds, then clustered in proper-motion space using K-means and Gaussian
Mixture Models to separate the cluster members from the foreground and
background field-star populations.

The most populated cluster is then standardised and analysed with
Principal Component Analysis to identify the dominant directions of
variance across the full feature set, and the resulting members are
plotted in a Hertzsprung-Russell-like diagram using Gaia BP, RP and G
photometry.

Data:
    m45-clean.csv  -- expected in a ``data/`` folder alongside the script.
                      A cleaned Gaia DR3 query toward the Pleiades.
"""

# **Imports**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# **Part 1**

# In[4]:


# Part 1: Reading and filtering the data
# Read the CSV file containing star data
df = pd.read_csv("data/m45-clean.csv")

# b) Plot proper motion characteristics
# Create a scatter plot of proper motion in right ascension vs declination
plt.figure(figsize=(10, 6))
plt.scatter(df['pmra'], df['pmdec'], s=1, alpha=0.5)
plt.xlabel('Proper Motion in Right Ascension (pmra) [mas/yr]')
plt.ylabel('Proper Motion in Declination (pmdec) [mas/yr]')
plt.title('Proper Motion Scatter Plot (Unfiltered Data)')
plt.grid()
plt.show()

# c) Filter outliers based on physical constraints
# Apply filters to remove unrealistic data points:
# - Only positive parallaxes
# - Relative parallax error < 20%
# - Proper motion within reasonable bounds (-55 to 55 mas/yr)
filtered_df = df[
    (df['parallax'] > 0) &  # Only positive parallaxes
    (df['parallax_error'] / df['parallax'] < 0.2) &  # Relative error < 20%
    (df['pmra'].between(-55, 55)) &  # Reasonable pmra range
    (df['pmdec'].between(-55, 55))  # Reasonable pmdec range
].copy()

# d) Replot filtered data
# Show the proper motion plot after filtering
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['pmra'], filtered_df['pmdec'], s=1, alpha=0.5)
plt.xlabel('Proper Motion in Right Ascension (pmra) [mas/yr]')
plt.ylabel('Proper Motion in Declination (pmdec) [mas/yr]')
plt.title('Proper Motion Scatter Plot (Filtered Data)')
plt.grid()
plt.show()

# e) Drop NaNs
# Remove any rows with missing values
filtered_df = filtered_df.dropna()
print(f"Number of stars after filtering: {len(filtered_df)}")


# **Part 2**

# In[6]:


# Part 2: Standardising the data
# Standardize all numerical columns to have mean=0 and std=1
scaler = StandardScaler()
columns_to_scale = [col for col in filtered_df.columns if filtered_df[col].dtype in ['float64', 'int64']]
scaled_df = filtered_df.copy()
scaled_df[columns_to_scale] = scaler.fit_transform(filtered_df[columns_to_scale])


# **Part 3**

# In[8]:


# Part 3: Clustering
# Extract proper motion data for clustering
pm_data = scaled_df[['pmra', 'pmdec']].values

# a) K-means clustering (original units)
# Perform K-means clustering with 2 clusters on proper motion data
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(pm_data)

# Plot K-means results in original units
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['pmra'], filtered_df['pmdec'], c=kmeans_labels, s=1, cmap='viridis')
plt.xlabel('Proper Motion in Right Ascension (pmra) [mas/yr]')
plt.ylabel('Proper Motion in Declination (pmdec) [mas/yr]')
plt.title('K-Means Clustering (n_clusters=2) - Original Units')
plt.grid()
plt.show()

# a) K-means clustering (standardised units)
# Plot K-means results in standardised units
plt.figure(figsize=(10, 6))
plt.scatter(scaled_df['pmra'], scaled_df['pmdec'], c=kmeans_labels, s=1, cmap='viridis')
plt.xlabel('Standardised Proper Motion in Right Ascension (pmra)')
plt.ylabel('Standardised Proper Motion in Declination (pmdec)')
plt.title('K-Means Clustering (n_clusters=2) - Standardized Data')
plt.grid()
plt.show()

# b) Gaussian Mixture Model clustering with 3 components (original units)
# Perform GMM clustering with 3 components
gmm3 = GaussianMixture(n_components=3, random_state=42)
gmm3_labels = gmm3.fit_predict(pm_data)

# Plot GMM results in original units
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['pmra'], filtered_df['pmdec'], c=gmm3_labels, s=1, cmap='viridis')
plt.xlabel('Proper Motion in Right Ascension (pmra) [mas/yr]')
plt.ylabel('Proper Motion in Declination (pmdec) [mas/yr]')
plt.title('GMM Clustering (n_components=3) - Original Units')
plt.xlim(-55, 55)
plt.ylim(-55, 55)
plt.grid()
plt.show()

# b) Gaussian Mixture Model clustering with 3 components (standardised units)
# Plot GMM results in standardised units
plt.figure(figsize=(10, 6))
plt.scatter(scaled_df['pmra'], scaled_df['pmdec'], c=gmm3_labels, s=1, cmap='viridis')
plt.xlabel('Standardized Proper Motion in Right Ascension (pmra)')
plt.ylabel('Standardized Proper Motion in Declination (pmdec)')
plt.title('GMM Clustering (n_components=3) - Standardized Data')
plt.grid()
plt.show()

# c) Optimised Gaussian Mixture Model clustering with 5 components (original units)
# Perform GMM clustering with 5 components
gmm5 = GaussianMixture(n_components=5, random_state=42)
gmm5_labels = gmm5.fit_predict(pm_data)

# Plot GMM results with 5 clusters in original units
plt.figure(figsize=(10, 6))
plt.scatter(filtered_df['pmra'], filtered_df['pmdec'], c=gmm5_labels, s=1, cmap='viridis')
plt.xlabel('Proper Motion in Right Ascension (pmra) [mas/yr]')
plt.ylabel('Proper Motion in Declination (pmdec) [mas/yr]')
plt.title('Optimised GMM Clustering (n_components=5) - Original Units')
plt.xlim(-55, 55)
plt.ylim(-55, 55)
plt.grid()
plt.show()

# c) Optimised Gaussian Mixture Model clustering with 5 components (standardised units)
# Plot GMM results with 5 clusters in standardised units
plt.figure(figsize=(10, 6))
plt.scatter(scaled_df['pmra'], scaled_df['pmdec'], c=gmm5_labels, s=1, cmap='viridis')
plt.xlabel('Standardised Proper Motion in Right Ascension (pmra)')
plt.ylabel('Standardised Proper Motion in Declination (pmdec)')
plt.title('Optimised GMM Clustering (n_components=5) - Standardized Data')
plt.grid()
plt.show()

# Calculate the centres of the 5 clusters identified by GMM
cluster_centers = []
for i in range(5):
    mask = gmm5_labels == i
    center_ra = filtered_df['pmra'][mask].mean()
    center_dec = filtered_df['pmdec'][mask].mean()
    cluster_centers.append((center_ra, center_dec))

# Identify the cluster in the bottom-right region of the plot
bottom_right_idx = np.argmax([ra - dec for ra, dec in cluster_centers])

# Print details about each cluster, including centre coordinates and star count
print("\nCluster Analysis (5 components):")
for i, (ra, dec) in enumerate(cluster_centers):
    status = "(Bottom-right)" if i == bottom_right_idx else ""
    print(f"Cluster {i}: {ra:.2f} mas/yr RA, {dec:.2f} mas/yr Dec, {np.sum(gmm5_labels == i)} stars {status}")

# Identify the smallest cluster as the interesting cluster for further analysis
cluster_counts = pd.Series(gmm5_labels).value_counts()
interesting_cluster = cluster_counts.idxmin()

# Extract stars belonging to the interesting cluster
interesting_stars = filtered_df[gmm5_labels == interesting_cluster].copy()
print(f"\nNumber of stars in interesting cluster: {len(interesting_stars)}")


# **Part 4**

# In[10]:


# Part 4: Principal Component Analysis
# Standardise the interesting cluster data
scaler_interesting = StandardScaler()
interesting_scaled = interesting_stars.copy()
interesting_scaled[columns_to_scale] = scaler_interesting.fit_transform(interesting_stars[columns_to_scale])

# Perform PCA on the interesting cluster
pca = PCA()
pca.fit(interesting_scaled[columns_to_scale])

# Get PCA results
components = pca.components_
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Print explained variance ratios
print("\nExplained variance ratio:")
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)")

# Calculate and print cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print("\nCumulative explained variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"PC1 to PC{i+1}: {cum_var:.3f} ({cum_var*100:.1f}%)")

# Combined improved visualisation of explained variance
plt.figure(figsize=(10, 6))

# Individual explained variance bars (green)
bars = plt.bar(range(1, len(explained_variance_ratio)+1), 
              explained_variance_ratio,
              alpha=0.5,
              color='green',
              label='Individual explained variance')

# Cumulative explained variance line (blue)
line = plt.plot(range(1, len(cumulative_variance)+1),
               cumulative_variance,
               '--o',
               color='blue',
               markersize=8,
               linewidth=2,
               label='Cumulative explained variance')

plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('PCA Explained Variance', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 1.05)
plt.xticks(range(1, len(explained_variance_ratio)+1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.show()


# **Part 5**

# In[12]:


# Part 5: Projection and factor analysis
# Project the data onto principal components
projected_data = pca.transform(interesting_scaled[columns_to_scale])

# Plot the projection onto the first two principal components
plt.figure(figsize=(10, 6))
plt.scatter(projected_data[:, 0], projected_data[:, 1], s=10, alpha=0.5)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}%)', fontsize=12)
plt.title('Projection onto Principal Components', fontsize=14)
plt.grid(True)
plt.show()

# Calculate loadings (how much each original feature contributes to each PC)
loadings_pc1 = components[0] * np.sqrt(explained_variance[0])
loadings_pc2 = components[1] * np.sqrt(explained_variance[1])

# Create a DataFrame of feature loadings
loadings_df = pd.DataFrame({
    'Feature': columns_to_scale,
    'PC1_loading': loadings_pc1,
    'PC2_loading': loadings_pc2
})

# Print top features contributing to PC1 and PC2
print("\nTop features for PC1:")
print(loadings_df.sort_values('PC1_loading', key=abs, ascending=False).head(3))

print("\nTop features for PC2:")
print(loadings_df.sort_values('PC2_loading', key=abs, ascending=False).head(3))

# Create the Hertzsprung-Russell-like diagram
# Plot color index (bp-rp) vs G-band magnitude
plt.figure(figsize=(10, 6))
plt.scatter(
    interesting_stars['bp'] - interesting_stars['rp'],  # bp-rp color index
    interesting_stars['g'],  # G-band magnitude
    s=10, alpha=0.5
)
plt.xlabel('bp - rp (Color Index)', fontsize=12)
plt.ylabel('G-band magnitude', fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis as smaller magnitude = brighter stars
plt.title('Hertzsprung-Russell-like Diagram', fontsize=14)
plt.grid(True)
plt.show()

