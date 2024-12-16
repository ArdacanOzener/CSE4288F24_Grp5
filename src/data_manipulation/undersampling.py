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
def cluster_based_undersampling(features, target, majority_class, n_clusters=7000):
    # Split majority and minority classes
    majority_features = features[target == majority_class]
    minority_features = features[target != majority_class]
    minority_target = target[target != majority_class]

    # Apply clustering on the majority class
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(majority_features)

    # Select one sample per cluster (closest to the centroid)
    cluster_centers = kmeans.cluster_centers_
    selected_indices = []
    for center in cluster_centers:
        distances = ((majority_features - center) ** 2).sum(axis=1)
        selected_indices.append(distances.idxmin())

    # Combine minority class and selected majority samples
    undersampled_features = pd.concat([minority_features, features.loc[selected_indices]])
    undersampled_target = pd.concat([minority_target, target.loc[selected_indices]])

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
