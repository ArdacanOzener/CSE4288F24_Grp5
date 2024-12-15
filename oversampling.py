import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# Load dataset
file_name = ["test", "ID_numeric"]
data_path = f'./datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}.csv'
save_path = f'./datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}'
data = pd.read_csv(data_path)

missing_values = data.isna().sum()
print(missing_values)

# Separate features and target
X_train = data.drop('loan_status', axis=1)  # Replace 'target' with your target column name
y_train = data['loan_status']

# Check class distribution
print("Original class distribution:", Counter(y_train), flush=True)


# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", Counter(y_train_smote))

# Apply ADASYN
adasyn = ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
print("Class distribution after ADASYN:", Counter(y_train_adasyn))

# Save the balanced datasets (optional)
smote_balanced = pd.concat([pd.DataFrame(X_train_smote), pd.DataFrame(y_train_smote, columns=['target'])], axis=1)
adasyn_balanced = pd.concat([pd.DataFrame(X_train_adasyn), pd.DataFrame(y_train_adasyn, columns=['target'])], axis=1)

smote_balanced.to_csv(save_path+"_smote.csv", index=False)
adasyn_balanced.to_csv(save_path+"_adasyn.csv", index=False)
