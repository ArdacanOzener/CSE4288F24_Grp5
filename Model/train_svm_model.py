import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import seaborn as sns
import time


# Load datasets
train_data = pd.read_csv('../datasets/training/training_data_ID_numeric.csv')
train_data_smote = pd.read_csv('../datasets/training/training_data_ID_numeric_smote.csv')
train_data_adasyn = pd.read_csv('../datasets/training/training_data_ID_numeric_adasyn.csv')
train_data_nearmiss = pd.read_csv('../datasets/training/training_data_ID_numeric_nearmiss.csv')
train_data_clusterbased = pd.read_csv('../datasets/training/training_data_ID_numeric_clusterbased.csv')
validation_data = pd.read_csv('../datasets/validation/validation_data_ID_numeric.csv')
test_data = pd.read_csv('../datasets/test/test_data_ID_numeric.csv')
test_data_undersampled = pd.read_csv('../datasets/test/test_data_ID_undersampled_numeric.csv')
validation_data_undersampled = pd.read_csv('../datasets/validation/validation_data_ID_undersampled_numeric.csv')

# Assign weights based on loan_status for train_data_weights_assigned
train_data_weights_assigned = train_data.copy()


# Separate features and target variable for validation and test data
X_validation = validation_data.drop(columns=['ID', 'loan_status'])
y_validation = validation_data['loan_status']
X_test = test_data.drop(columns=['ID', 'loan_status'])
y_test = test_data['loan_status']

X_validation_undersampled = validation_data_undersampled.drop(columns=['ID', 'loan_status'])
y_validation_undersampled = validation_data_undersampled['loan_status']
X_test_undersampled = test_data_undersampled.drop(columns=['ID', 'loan_status'])
y_test_undersampled = test_data_undersampled['loan_status']

# Initialize dictionaries to store accuracies
validation_accuracies = {}
test_accuracies = {}

# Initialize total training time
total_training_time = 0

# Function to train and evaluate model with different kernels
def train_and_evaluate(train_data, label, X_val, y_val, X_tst, y_tst, kernel):
    global total_training_time
    start_time = time.time()
    
    X_train = train_data.drop(columns=['ID', 'loan_status'])
    y_train = train_data['loan_status']
    
    if(label == "original_weight" ):
        svm_model = SVC(kernel=kernel, class_weight= {0: 1, 1: 3.5})
    else: 
        svm_model = SVC(kernel=kernel)
    
    svm_model.fit(X_train, y_train)
    
    y_pred_validation = svm_model.predict(X_val)
    y_pred_test = svm_model.predict(X_tst)
    
    validation_accuracy = sum(y_val == y_pred_validation) / len(y_val)
    test_accuracy = sum(y_tst == y_pred_test) / len(y_tst)
    
    validation_accuracies[f"{label}_{kernel}"] = validation_accuracy
    test_accuracies[f"{label}_{kernel}"] = test_accuracy

    validation_precision = sum((y_val == 1) & (y_pred_validation == 1)) / sum(y_pred_validation == 1)
    validation_recall = sum((y_val == 1) & (y_pred_validation == 1)) / sum(y_val == 1)
    validation_f1 = 2 * (validation_precision * validation_recall) / (validation_precision + validation_recall)

    test_precision = sum((y_tst == 1) & (y_pred_test == 1)) / sum(y_pred_test == 1)
    test_recall = sum((y_tst == 1) & (y_pred_test == 1)) / sum(y_tst == 1)
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    
    end_time = time.time()
    training_time = end_time - start_time
    total_training_time += training_time
    
    print(f"\n{label.capitalize()}, Kernel: {kernel}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Validation Accuracy: {validation_accuracy:.2f}")
    '''
    print(f"Validation Precision: {validation_precision:.2f}")
    print(f"Validation Recall: {validation_recall:.2f}")
    print(f"Validation F1 Score: {validation_f1:.2f}")
    '''
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Precision: {test_precision:.2f}")
    print(f"Test Recall: {test_recall:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")


    conf_matrix = confusion_matrix(y_tst, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
# List of kernels to evaluate
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Train individual SVM models with different kernels
for kernel in kernels:
    train_and_evaluate(train_data, 'original', X_validation, y_validation, X_test, y_test, kernel)
    train_and_evaluate(train_data_smote, 'smote', X_validation, y_validation, X_test, y_test, kernel)
    train_and_evaluate(train_data_adasyn, 'adasyn', X_validation, y_validation, X_test, y_test, kernel)
    train_and_evaluate(train_data_nearmiss, 'nearmiss', X_validation, y_validation, X_test, y_test, kernel)
    train_and_evaluate(train_data_clusterbased, 'clusterbased', X_validation, y_validation, X_test, y_test, kernel)
    train_and_evaluate(train_data_weights_assigned, 'original_weight', X_validation, y_validation, X_test, y_test, kernel)

    # Train individual SVM models on undersampled data
    train_and_evaluate(train_data, 'original_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)
    train_and_evaluate(train_data_smote, 'smote_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)
    train_and_evaluate(train_data_adasyn, 'adasyn_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)
    train_and_evaluate(train_data_nearmiss, 'nearmiss_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)
    train_and_evaluate(train_data_clusterbased, 'clusterbased_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)


# Print total training time
print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

# Plot validation accuracies
plt.figure(figsize=(12, 6))
plt.bar(validation_accuracies.keys(), validation_accuracies.values())
plt.xlabel('Model and Kernel')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracies for Different Models and Kernels')
plt.xticks(rotation=90)
plt.show()


# Train individual SVM models on undersampled data
for kernel in kernels:
    train_and_evaluate(test_data_undersampled, 'undersampled_test', X_test_undersampled, y_test_undersampled, X_test, y_test, kernel)
    train_and_evaluate(validation_data_undersampled, 'undersampled_validation', X_validation_undersampled, y_validation_undersampled, X_test, y_test, kernel)


# Ensemble method using VotingClassifier with different kernels
def train_and_evaluate_ensemble(X_val, y_val, X_tst, y_tst, kernel):
    X_train = train_data.drop(columns=['ID', 'loan_status'])
    y_train = train_data['loan_status']
    
    svm_original = SVC(kernel=kernel, probability=True)
    svm_smote = SVC(kernel=kernel, probability=True)
    svm_adasyn = SVC(kernel=kernel, probability=True)
    svm_nearmiss = SVC(kernel=kernel, probability=True)
    svm_clusterbased = SVC(kernel=kernel, probability=True)
    svm_weights_assigned = SVC(kernel=kernel, probability=True)
    
    ensemble_model = VotingClassifier(estimators=[
        ('original', svm_original),
        ('smote', svm_smote),
        ('adasyn', svm_adasyn),
        ('nearmiss', svm_nearmiss),
        ('clusterbased', svm_clusterbased),
        ('original_weight', svm_weights_assigned)
    ], voting='soft')
    
    ensemble_model.fit(X_train, y_train)
    
    y_pred_validation = ensemble_model.predict(X_val)
    y_pred_test = ensemble_model.predict(X_tst)
    
    validation_accuracy = sum(y_val == y_pred_validation) / len(y_val)
    test_accuracy = sum(y_tst == y_pred_test) / len(y_tst)
    
    validation_precision = precision_score(y_val, y_pred_validation)
    validation_recall = recall_score(y_val, y_pred_validation)
    validation_f1 = f1_score(y_val, y_pred_validation)
    
    test_precision = precision_score(y_tst, y_pred_test)
    test_recall = recall_score(y_tst, y_pred_test)
    test_f1 = f1_score(y_tst, y_pred_test)
    
    print(f"Ensemble Model (Kernel: {kernel}):")
    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\nValidation Precision:", validation_precision)
    print("Validation Recall:", validation_recall)
    print("Validation F1 Score:", validation_f1)
    print("\nTest Precision:", test_precision)
    print("Test Recall:", test_recall)
    print("Test F1 Score:", test_f1)

    
    conf_matrix = confusion_matrix(y_tst, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    

# Train ensemble model on original data with different kernels
for kernel in kernels:
    train_and_evaluate_ensemble(X_validation, y_validation, X_test, y_test, kernel)

# Train ensemble model on undersampled data with different kernels
for kernel in kernels:
    train_and_evaluate_ensemble(X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled, kernel)
