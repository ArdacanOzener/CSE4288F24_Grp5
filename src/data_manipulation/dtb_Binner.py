import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import time
import json


df = pd.read_csv("../../datasets/loan_data.csv")

columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length"]

class_label = "loan_status"

y = df[class_label]

feature = []


for data_column in columns:

    bin= []
    X = df[data_column].values.reshape(-1, 1)

    for i in range(2,11):
        results = []
        start_time = time.time()
        # Train a decision tree
        tree = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=i, random_state=42)
        tree.fit(X, y)

        end_time = time.time()

        impurities = tree.tree_.impurity

        # Get split points
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]


        results.append({"Thresholds": thresholds.tolist()})
        results.append({"Average of Entropies": sum(impurities)/len(impurities)})
        results.append({"Time": (end_time-start_time)})
        results.append({"Entropies": impurities.tolist()})

        bin.append({f"{i}": results})

    feature.append({f"{data_column}": bin})


file_path = "under_train_bin_analysis.json"


# Write the list of dictionaries to a JSON file
with open(file_path, "w") as json_file:
    json.dump(feature, json_file, indent=4)
