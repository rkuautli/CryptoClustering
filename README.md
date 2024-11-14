# CryptoClustering
## Cryptocurrency Market Segmentation Using K-Means Clustering and PCA

## Project Overview

This project applies unsupervised machine learning techniques to analyze and cluster cryptocurrencies based on their market data. The primary goal is to identify natural groupings of cryptocurrencies, allowing for deeper insights into market trends and relationships. The approach leverages the following steps:

Data Preprocessing: Includes normalization to standardize the dataset and ensure that all features contribute equally to the clustering model.
Optimal Cluster Determination: Utilizes the elbow method to find the ideal number of clusters (k) that minimizes the within-cluster variance while maintaining model simplicity.
Dimensionality Reduction: Applies Principal Component Analysis (PCA) to reduce the feature space, improving computational efficiency and potentially enhancing clustering performance by removing noise.
Cluster Visualization: Visualizes the results using scatter plots to intuitively represent how different cryptocurrencies are grouped based on market behavior.
The entire analysis will be conducted using Python and core data science libraries such as Pandas, scikit-learn, and hvPlot, providing a robust and scalable solution for cryptocurrency market segmentation.

## Methodology

## Step 1: Rename the Notebook
Rename the notebook from Crypto_Clustering_starter_code.ipynb to Crypto_Clustering.ipynb to standardize file naming conventions and prepare for the analysis.

## Step 2: Data Ingestion and Initial Exploration
Load Data: Begin by loading the crypto_market_data.csv dataset into a Pandas DataFrame for further examination.
Exploratory Data Analysis (EDA): Conduct an initial review of the data, including summary statistics (e.g., mean, median, standard deviation) and visualizations (e.g., histograms, scatter plots) to understand the distribution of features, detect potential outliers, and identify data imbalances or missing values.

## Step 3: Data Normalization
Feature Scaling: Given that K-means is sensitive to the scale of the features, normalize the data using StandardScaler from scikit-learn to standardize all features to have zero mean and unit variance. This step ensures that no single feature dominates the clustering process due to differing magnitudes.
Create a new DataFrame containing the normalized data, preserving the coin_id as the index for traceability.

## Step 4: Determining the Optimal Number of Clusters (k)
The Elbow Method is employed to identify the optimal number of clusters for the K-means algorithm:

Compute the inertia (sum of squared distances between samples and their assigned cluster centers) for a range of k values, from 1 to 11.
Plot the inertia against the number of clusters to visualize the "elbow point," where the rate of decrease in inertia slows down. The "elbow" represents the most suitable number of clusters, balancing model complexity and performance.
Document and justify the optimal k based on the observed elbow point.

## Step 5: K-Means Clustering
Model Initialization: Initialize the K-means algorithm with the optimal k identified in Step 4. Fit the model to the normalized data, and assign each cryptocurrency to its respective cluster.
Cluster Assignment: Append the predicted cluster labels to the scaled DataFrame as a new column, allowing for direct association of cryptocurrencies with their clusters.
Cluster Visualization: Visualize the clustering results using a scatter plot generated with hvPlot. Use price_change_percentage_24h for the x-axis and price_change_percentage_7d for the y-axis, with data points colored according to their respective clusters. This will provide an intuitive representation of how the cryptocurrencies are grouped based on their market behavior.

## Step 6: Dimensionality Reduction Using Principal Component Analysis (PCA)
PCA Implementation: Apply Principal Component Analysis (PCA) to the normalized data to reduce the dimensionality of the feature space. PCA will transform the data into a set of orthogonal components, ordered by the amount of variance they capture from the original data. By reducing the dataset to the top three principal components, we can simplify the clustering process without sacrificing too much information.
Explained Variance: Analyze and document the explained variance ratio for each principal component. This will provide insight into how much of the original data's variance is retained after transformation.
Create a new DataFrame with the PCA-transformed data, retaining the coin_id index to maintain cryptocurrency identifiers.

## Step 7: Reassessing the Optimal k Using PCA-Transformed Data
Repeat the Elbow Method on the PCA-reduced dataset to determine if dimensionality reduction affects the optimal number of clusters:

Calculate inertia values for a range of k values (from 1 to 11) using the PCA-reduced features.
Plot the elbow curve to identify the optimal k based on the PCA-transformed data and compare it to the k value obtained from the original, untransformed data.
Document the chosen k and analyze any differences in clustering behavior due to the reduction in feature space.

## Step 8: Final Clustering Using PCA-Transformed Data
Clustering with PCA: Using the optimal k determined from the PCA data, initialize and fit the K-means model on the PCA-reduced dataset.
Predict cluster labels and append them to the scaled PCA DataFrame.
Visualization: Plot the clustering results in a scatter plot, using the first two principal components (PCA1 and PCA2) as the axes, and color the points according to their cluster labels. Include the coin_id column for reference in the hover tool.
Reflect on the advantages and potential limitations of clustering in a reduced feature space and assess how dimensionality reduction impacted the overall clustering quality.

## Conclusion
This analysis provides a robust framework for clustering cryptocurrencies using K-means and Principal Component Analysis (PCA). Through careful preprocessing, dimensionality reduction, and cluster evaluation, this project demonstrates how unsupervised learning can be used to segment cryptocurrencies into meaningful groups based on market dynamics. Key takeaways include:

## K-means clustering: A versatile technique for partitioning datasets into distinct groups, useful for identifying patterns in complex, high-dimensional data.
Elbow Method: An effective technique for determining the optimal number of clusters by balancing model complexity with performance.
Principal Component Analysis (PCA): A powerful tool for reducing the dimensionality of data, improving computational efficiency, and mitigating noise in the clustering process.
Data Normalization: Essential for ensuring fair contribution from each feature in distance-based algorithms like K-means.
