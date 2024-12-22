import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time
import json

# Load dataset
df = pd.read_csv("../../datasets/loan_data.csv")

columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length"]

class_label = "loan_status"

y = df[class_label]

feature_results = []

for data_column in columns:

    bin_results = []
    X = df[data_column].values.reshape(-1, 1)

    for i in range(2, 21):
        results = []
        start_time = time.time()

        # Train a decision tree
        tree = DecisionTreeClassifier(class_weight={0: 1, 1: 3.5}, criterion='entropy', max_leaf_nodes=i, random_state=42)
        tree.fit(X, y)

        end_time = time.time()

        # Identify leaf nodes
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        impurities = tree.tree_.impurity

        leaf_indices = np.where((children_left == -1) & (children_right == -1))[0]

        # Calculate average entropy of leaf nodes
        leaf_entropies = impurities[leaf_indices]
        average_leaf_entropy = np.mean(leaf_entropies)

        # Get split points
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]

        results.append({"Thresholds": thresholds.tolist()})
        results.append({"Average Leaf Entropy": average_leaf_entropy})
        results.append({"Time": end_time - start_time})
        results.append({"Leaf Entropies": leaf_entropies.tolist()})

        bin_results.append({f"max_leaf_nodes={i}": results})

    feature_results.append({data_column: bin_results})

# Save results to JSON file
file_path = "bin_analysis.json"
with open(file_path, "w") as json_file:
    json.dump(feature_results, json_file, indent=4)
