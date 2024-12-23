import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv('datasets/training/training_data_ID_numeric.csv')
train_data_smote = pd.read_csv('datasets/training/training_data_ID_numeric_smote.csv')
train_data_adasyn = pd.read_csv('datasets/training/training_data_ID_numeric_adasyn.csv')
train_data_nearmiss = pd.read_csv('datasets/training/training_data_ID_numeric_nearmiss.csv')
train_data_clusterbased = pd.read_csv('datasets/training/training_data_ID_numeric_clusterbased.csv')
train_data_weighted = pd.read_csv('datasets/training/training_data_ID_numeric.csv')

validation_data = pd.read_csv('datasets/validation/validation_data_ID_numeric.csv')
test_data = pd.read_csv('datasets/test/test_data_ID_numeric.csv')

# Initialize dictionaries to store accuracies
validation_accuracies = {}
test_accuracies = {}

def train_and_evaluate_model(training_data, label):
    # Prepare the data
    X_train = training_data.drop(['ID', 'loan_status'], axis=1)
    y_train = training_data['loan_status']
    
    X_test = test_data.drop(['ID', 'loan_status'], axis=1)
    y_test = test_data['loan_status']
    
    X_validation = validation_data.drop(['ID', 'loan_status'], axis=1)
    y_validation = validation_data['loan_status']

    # Initialize and train the model
    if label == "original_weighted":
        # Use class weights for imbalanced original dataset
        model = LogisticRegression(max_iter=1000, class_weight={0: 1, 1: 3.5})
    else:
        model = LogisticRegression(max_iter=1000)
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred_validation = model.predict(X_validation)
    
    # Calculate accuracies
    validation_accuracy = accuracy_score(y_validation, y_pred_validation)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    validation_accuracies[label] = validation_accuracy
    test_accuracies[label] = test_accuracy
    
    # Print results
    print(f"\n{label.capitalize()} Dataset Results:")
    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\nValidation Classification Report:")
    print(classification_report(y_validation, y_pred_validation))
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {label.capitalize()}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)

# Train and evaluate models with different datasets
train_and_evaluate_model(train_data, "original")
train_and_evaluate_model(train_data_smote, "smote")
train_and_evaluate_model(train_data_adasyn, "adasyn")
train_and_evaluate_model(train_data_nearmiss, "nearmiss")
train_and_evaluate_model(train_data_clusterbased, "clusterbased")
train_and_evaluate_model(train_data_weighted, "original_weighted")

# Print comparison of accuracies
print("\nAccuracy Comparison:")
print("\nValidation Accuracies:")
for dataset, accuracy in validation_accuracies.items():
    print(f"{dataset.capitalize()}: {accuracy:.4f}")

print("\nTest Accuracies:")
for dataset, accuracy in test_accuracies.items():
    print(f"{dataset.capitalize()}: {accuracy:.4f}")
