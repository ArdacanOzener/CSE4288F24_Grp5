# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 23:43:08 2024

@author: ecegn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri yükle
data = pd.read_csv("binned_loan_data.csv")

# Binning yapılmış sütunlar
bin_columns = [
    "person_age_binned", 
    "person_income_binned", 
    "person_emp_exp_binned", 
    "loan_amnt_binned", 
    "loan_int_rate_binned", 
    "loan_percent_income_binned", 
    "cb_person_cred_hist_length_binned"
]

class_label_column = "loan_status"  

# Her bindeki loan_status dağılımını incele
for column in bin_columns:
    
    bin_counts = data.groupby([column, class_label_column]).size().unstack(fill_value=0)
    
    #görsel
    bin_counts.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'salmon'])
    plt.title(f"{column.replace('_', ' ').title()} - Loan Status Distribution", fontsize=14)
    plt.xlabel(f"{column.replace('_', ' ').title()}", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
