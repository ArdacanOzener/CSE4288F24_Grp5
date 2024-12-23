import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def random_undersampling(data_path, save_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Separate features and target
    X = data.drop(['loan_status', 'ID'], axis=1)  # Exclude 'ID' column
    y = data['loan_status']
    ids = data['ID']  # Save the ID column separately

    # Check class distribution
    print("Original class distribution:", Counter(y), flush=True)

    # Apply Random Undersampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Allow duplication to ensure balance
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name='loan_status')

    # Restore the ID column
    resampled_data = pd.concat([ids.loc[X_resampled.index], X_resampled, y_resampled], axis=1)

    # Save the balanced dataset
    resampled_data.to_csv(save_path, index=False)

    # Check new class distribution
    print("Class distribution after Random Undersampling:", Counter(y_resampled), flush=True)

# Example usage for test_data and validation_data
random_undersampling('../../datasets/test/test_data_ID.csv', '../../datasets/test/test_data_ID_undersampled.csv')
random_undersampling('../../datasets/validation/validation_data_ID.csv', '../../datasets/validation/validation_data_ID_undersampled.csv')