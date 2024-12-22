import pandas as pd
from sklearn.cluster import KMeans
from imblearn.under_sampling import NearMiss


# Load dataset
file_name = ["training", "ID_numeric"]
data_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}.csv'
save_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}'
data = pd.read_csv(data_path)


# Separate features and target
features = data.drop(['loan_status', 'ID'], axis=1)  # Exclude 'ID' column
target = data['loan_status']
ids = data['ID']  # Save the ID column separately

# Check class distribution
print("Original Class Distribution:")
print(target.value_counts())


# --- Cluster-Based Undersampling ---
def cluster_based_undersampling(features, target, majority_class, n_clusters=10, n_samples_per_cluster=700):
    # Split majority and minority classes
    majority_features = features[target == majority_class]
    minority_features = features[target != majority_class]
    minority_target = target[target != majority_class]

    

    # Apply clustering on the majority class
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(majority_features)

    # Group majority instances by cluster
    cluster_labels = kmeans.labels_
    majority_features = majority_features.reset_index(drop=True)
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_indices[label].append(idx)

    # Select closest samples from each cluster
    all_selected_indices = []
    leftover_indices_per_cluster = []
    cluster_centers = kmeans.cluster_centers_

    for cluster_id, indices in cluster_indices.items():
        cluster_data = majority_features.iloc[indices]
        center = cluster_centers[cluster_id]

        # Compute distances from cluster center
        distances = ((cluster_data - center) ** 2).sum(axis=1)
        sorted_indices = distances.argsort()


        # Select closest n_samples_per_cluster or all available if less
        selected = sorted_indices[:n_samples_per_cluster]
        all_selected_indices.extend(cluster_data.index[selected])

        # Track leftover indices if cluster doesn't meet the target
        leftover_indices = sorted_indices[n_samples_per_cluster:]
        leftover_indices_per_cluster.append(cluster_data.index[leftover_indices])

    # Calculate deficit and distribute it among clusters
    deficit = n_clusters * n_samples_per_cluster - len(all_selected_indices)
    if deficit > 0:
        per_cluster_deficit = deficit // n_clusters
        extra = deficit % n_clusters

        for i, leftover_indices in enumerate(leftover_indices_per_cluster):
            # Contribute equally and handle the extra deficit
            num_to_select = per_cluster_deficit + (1 if i < extra else 0)
            all_selected_indices.extend(leftover_indices[:num_to_select])

    # Combine minority and selected majority samples
    selected_majority_features = majority_features.loc[all_selected_indices]

    selected_majority_target = target.iloc[selected_majority_features.index] 
    selected_majority_target[:] = 0

    undersampled_features = pd.concat([minority_features, selected_majority_features])
    undersampled_target = pd.concat([minority_target, selected_majority_target])


    return undersampled_features, undersampled_target


# Apply Cluster-Based Undersampling
majority_class = target.value_counts().idxmax()
cluster_features, cluster_target = cluster_based_undersampling(features, target, majority_class)

# --- NearMiss Undersampling ---
# Combine features and target for NearMiss
data_combined = pd.concat([features, target], axis=1)
nearmiss = NearMiss(version=1)  # Choose version 1, 2, or 3 as needed
nm_features, nm_target = nearmiss.fit_resample(features, target)

# Restore the ID column for both methods
cluster_result = pd.concat([ids.loc[cluster_features.index], cluster_features, cluster_target], axis=1)
nearmiss_result = pd.concat([ids.loc[nm_features.index], pd.DataFrame(nm_features, columns=features.columns), nm_target], axis=1)

# Save the results
cluster_result.to_csv(save_path+"_clusterbased.csv", index=False)
nearmiss_result.to_csv(save_path+"_nearmiss.csv", index=False)

# Check new class distributions
print("Cluster-Based Undersampling Class Distribution:")
print(cluster_target.value_counts())
print("NearMiss Class Distribution:")
print(pd.Series(nm_target).value_counts())
