import pandas as pd
data_path = 'loan_data.csv'
save_path = 'loan_data_numerical.csv'
df = pd.read_csv(data_path)

#person gender
#male, female
#convert to female = 0, male = 1

gender_mapping = {'female': 0, 'male': 1}
df['person_gender'] = df['person_gender'].map(gender_mapping)

#person education
#High School, Associate, Bachelor, Master, Doctor
#High School = 0, Associate = 0.25, Bachelor = 0.5, Master = 0.75, Doctor = 1
education_mapping = {'High School': 0, 'Associate': 0.25, 'Bachelor': 0.5, 'Master': 0.75, 'Doctor': 1}
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
    'loan_amnt', 
    'loan_int_rate', 
    'loan_percent_income', 
    'cb_person_cred_hist_length', 
    'credit_score'
]

# Min-Max Normalizasyonu
df[columns_to_normalize] = df[columns_to_normalize].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

#save to csv
df.to_csv(save_path, index=False)


