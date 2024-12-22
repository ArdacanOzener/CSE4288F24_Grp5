import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
file_path = "../../datasets/training/training_data_ID_numeric.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Filter data to include only instances with 'loan_status' equal to 0
filtered_data = data[data['loan_status'] == 0]

# Extract features (replace 'feature_columns' with your actual feature column names)
feature_columns = [col for col in filtered_data.columns if col != ('loan_status' or 'ID')]
X = filtered_data[feature_columns].values

# Compute WCSS for different values of k
wcss = []
number = 7000
k_values = [i for i in range(1, number + 1) if number % i == 0]


for k in k_values: 
    print(k, flush=True)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_values, wcss, marker='o', linestyle='-', color='b')
plt.xscale('log')  # Use logarithmic scale for better visualization
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('WCSS vs. Number of Clusters (k)')
plt.grid(True)
plt.savefig("elbow_plot.png")  # Save the figure

# Optionally, print the WCSS values
for k, w in zip(k_values, wcss):
    print(f"k={k}, WCSS={w:.2f}")
