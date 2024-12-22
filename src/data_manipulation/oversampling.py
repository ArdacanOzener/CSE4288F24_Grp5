import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter
import hashlib

# Load dataset
file_name = ["training", "ID_numeric"]
data_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}.csv'
save_path = f'../../datasets/{file_name[0]}/{file_name[0]}_data_{file_name[1]}'
data = pd.read_csv(data_path)


# Separate features and target
X_train = data.drop(['loan_status', 'ID'], axis=1)  # Exclude 'ID' column
y_train = data['loan_status']
ids = data['ID']  # Save the ID column separately

# Check class distribution
print("Original class distribution:", Counter(y_train), flush=True)

# Function to generate consistent synthetic IDs
def generate_synthetic_ids(X_synthetic, starting_id=50000):
    id_map = {}
    synthetic_ids = []
    current_id = starting_id

    for _, row in X_synthetic.iterrows():
        # Create a unique hash for each row
        row_hash = hashlib.md5(row.to_string().encode()).hexdigest()
        if row_hash not in id_map:
            id_map[row_hash] = current_id
            current_id += 1
        synthetic_ids.append(id_map[row_hash])
    
    return synthetic_ids, current_id

# Function to force exact balance by oversampling
def force_balance(X, y, majority_class, minority_class):
    class_counts = Counter(y)
    max_count = class_counts[majority_class]
    diff = max_count - class_counts[minority_class]

    if diff > 0:
        minority_indices = y[y == minority_class].index
        extra_samples = X.loc[minority_indices].sample(diff, replace=True, random_state=42)
        X_balanced = pd.concat([X, extra_samples])
        y_balanced = pd.concat([y, pd.Series([minority_class] * diff, index=extra_samples.index)])
        return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)
    elif diff < 0:
        X_balanced = X[:2*(max_count)]
        y_balanced = y[:2*(max_count)]
        return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True)
    return X, y

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Allow duplication to ensure balance
X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
y_train_smote = pd.Series(y_train_smote, name='loan_status')
X_train_smote, y_train_smote = force_balance(X_train_smote, y_train_smote, majority_class=0, minority_class=1)

# Assign IDs to SMOTE synthetic samples
synthetic_ids_smote, last_id = generate_synthetic_ids(X_train_smote[len(X_train):], starting_id=50000)
smote_ids = list(ids) + synthetic_ids_smote

print("Class distribution after SMOTE:", Counter(y_train_smote))

# Apply ADASYN
adasyn = ADASYN(random_state=42, sampling_strategy=1.0)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)


# Allow duplication to ensure balance
X_train_adasyn = pd.DataFrame(X_train_adasyn, columns=X_train.columns)
y_train_adasyn = pd.Series(y_train_adasyn, name='loan_status')
print("Class distribution after ADASYN:", Counter(y_train_adasyn))
X_train_adasyn, y_train_adasyn = force_balance(X_train_adasyn, y_train_adasyn, majority_class=0, minority_class=1)

# Assign IDs to ADASYN synthetic samples
synthetic_ids_adasyn, _ = generate_synthetic_ids(X_train_adasyn[len(X_train):], starting_id=last_id)
adasyn_ids = list(ids) + synthetic_ids_adasyn

print("Class distribution after ADASYN:", Counter(y_train_adasyn))

# Save the balanced datasets
smote_balanced = pd.concat([
    pd.Series(smote_ids, name='ID'),
    X_train_smote,
    y_train_smote
], axis=1)

adasyn_balanced = pd.concat([
    pd.Series(adasyn_ids, name='ID'),
    X_train_adasyn,
    y_train_adasyn
], axis=1)

smote_balanced.to_csv(save_path+"_smote.csv", index=False)
adasyn_balanced.to_csv(save_path+"_adasyn.csv", index=False)
