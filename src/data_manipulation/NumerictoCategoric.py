from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np

# dataset yükle
file_name = ["test", "ID"]
data_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}.csv'
save_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}_categoric.csv'
data = pd.read_csv(data_path)


columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length"]


class_label = "loan_status"

def calculate_information_gain(data, columns, target):
    ig_values = {}
    for column in columns:
        ig = mutual_info_classif(data[[column]], data[target], discrete_features=False)
        ig_values[column] = ig[0]
    return ig_values


def determine_bins_by_ig(ig_values, min_bins=3, max_bins=10):
    min_ig = min(ig_values.values())
    max_ig = max(ig_values.values())
    
    #normalize et
    normalized_ig = {}
    for col, ig in ig_values.items():
        normalized_ig[col] = (ig - min_ig) / (max_ig - min_ig)
    
    # Normalize edilen bilgi kazancına göre bin sayısını belirle
    bins_by_ig = {}
    for col, norm_ig in normalized_ig.items():
        bins_by_ig[col] = int(min_bins + (max_bins - min_bins) * norm_ig)
    
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

#Information gain
ig_values = calculate_information_gain(data, columns, class_label)

#Information gaine göre bin belirle
dynamic_bins = determine_bins_by_ig(ig_values)

for column in columns:
    apply_binning(column, n_bins=dynamic_bins[column])


data.to_csv(save_path, index=False)

