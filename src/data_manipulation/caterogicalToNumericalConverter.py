import pandas as pd
def normalize_dataframe(df, column_min_max_values, columns_to_normalize):
    for column in columns_to_normalize:
        if column in column_min_max_values:
            min_value = column_min_max_values[column]['min']
            max_value = column_min_max_values[column]['max']
            df[column] = (df[column] - min_value) / (max_value - min_value)
    return df

data_path = '../../datasets/validation/validation_data_ID_undersampled.csv'
save_path = '../../datasets/validation/validation_data_ID_undersampled_numeric.csv'
df = pd.read_csv(data_path)

#person gender
#male, female
#convert to female = 0, male = 1

gender_mapping = {'female': 0, 'male': 1}
df['person_gender'] = df['person_gender'].map(gender_mapping)

#person education
#High School, Associate, Bachelor, Master, Doctor
#High School = 0, Associate = 0.25, Bachelor = 0.5, Master = 0.75, Doctor = 1
education_mapping = {'High School': 0, 'Associate': 0.25, 'Bachelor': 0.5, 'Master': 0.75, 'Doctorate': 1}
df['person_education'] = df['person_education'].map(education_mapping)

#person home ownership
#RENT, OWN, MORTGAGE, OTHER
#one hot encoding
df = pd.get_dummies(df, columns=['person_home_ownership'], prefix='home', dtype=int)

#loan intent
#PERSONAL, EDUCATION, MEDICAL, VENTURE, DEBTCONSOLIDATION
#one hot encoding
df = pd.get_dummies(df, columns=['loan_intent'], prefix='intent', dtype=int)

#previous loan defaults on file
loan_default_mapping = {'No': 0, 'Yes': 1}
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map(loan_default_mapping)

#normalize
columns_to_normalize = [
    'person_age', 
    'person_income',
    'person_emp_exp',
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_cred_hist_length', 
    'credit_score'
]

min_max_values_path = '../../config_files/column_min_max_values.json'
min_max_values = pd.read_json(min_max_values_path)
df = normalize_dataframe(df, min_max_values, columns_to_normalize)

#move target feature to the end
target_feature = 'loan_status'
target_feature_index = df.columns.get_loc(target_feature)
df = df[[col for col in df.columns if col != target_feature] + [target_feature]]

#save to csv
df.to_csv(save_path, index=False)







