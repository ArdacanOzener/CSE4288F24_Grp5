from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

# Veri yükle
data = pd.read_csv("test_data_ID.csv")

columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length"]

class_label = "loan_status"

#loan_statusa göre information gain hesapla
def calculate_inf_gain(data, columns, target):
    ig_values = {}
    for column in columns:
        ig = mutual_info_classif(data[[column]], data[target], discrete_features=False)
        ig_values[column] = ig[0]
    return ig_values

#bin hesapla, min 3 max 10
def determine_bins(ig_values, min_bins=3, max_bins=10):
    min_ig = min(ig_values.values())
    max_ig = max(ig_values.values())
    
    # Normalizasyon ve ters çevirme
    reversed_ig = {col: max_ig - ig for col, ig in ig_values.items()} 

    # Normalize et
    min_reversed_ig = min(reversed_ig.values())
    max_reversed_ig = max(reversed_ig.values())
    normalized_ig = {col: (rev_ig - min_reversed_ig) / (max_reversed_ig - min_reversed_ig) for col, rev_ig in reversed_ig.items()}
    
    # Information gaine göre bin hesapla
    bins_by_ig = {col: int(min_bins + (max_bins - min_bins) * norm_ig) for col, norm_ig in normalized_ig.items()}
    
    return bins_by_ig

# Binning fonksiyonu
def apply_binning(column_name, n_bins, strategy='quantile'):
    data_column = np.array(data[column_name]).reshape(-1, 1)
    
    # KBinsDiscretizer ile binleme
    binning_model = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    binning_model.fit(data_column)
    
    bin_edges = binning_model.bin_edges_[0]
    bin_labels_list = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
    
    data[f"{column_name}_binned"] = data[column_name].apply(
        lambda x: bin_labels_list[binning_model.transform([[x]])[0][0].astype(int)]
    )
    
    return bin_labels_list

# Information gain hesapla
ig_values = calculate_inf_gain(data, columns, class_label)

# Information gaine göre ters bin belirle
bins= determine_bins(ig_values)

for column in columns:
    apply_binning(column, n_bins=bins[column])

data.to_csv("test_data_ID_categoric.csv", index=False)
