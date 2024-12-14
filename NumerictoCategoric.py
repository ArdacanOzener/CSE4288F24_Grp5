from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("loan_data.csv")

# Binning yapılacak sütunlar
columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length"]

def apply_binning(column_name, n_bins=4, strategy='quantile'):
   
    data_column = np.array(data[column_name]).reshape(-1, 1)
    
    # KBinsDiscretizer ile binleme
    binning_model = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    binning_model.fit(data_column)
    

    bin_edges = binning_model.bin_edges_[0]
    
    
    bin_labels_list = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)]
    
    # Bin etiketlerini uygula
    data[f"{column_name}_binned"] = data[column_name].apply(
        lambda x: bin_labels_list[binning_model.transform([[x]])[0][0].astype(int)]  # Dönüşümün ilk değerini al
    )
    
    return bin_labels_list

# Her bir sütun için binning işlemi yap
for column in columns:
    # Binning işlemi ve bin etiketlerinin uygulanması
    apply_binning(column)
    

data.to_csv("binned_loan_data.csv", index=False)


# Tüm sütun isimlerini yazdır
print(data.columns.tolist())
print(data.head())
