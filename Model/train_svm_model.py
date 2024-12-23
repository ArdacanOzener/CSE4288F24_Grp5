import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
import seaborn as sns


# Load datasets
train_data = pd.read_csv('../datasets/training/training_data_ID_numeric.csv')
train_data_smote = pd.read_csv('../datasets/training/training_data_ID_numeric_smote.csv')
train_data_adasyn = pd.read_csv('../datasets/training/training_data_ID_numeric_adasyn.csv')
train_data_nearmiss = pd.read_csv('../datasets/training/training_data_ID_numeric_nearmiss.csv')
train_data_clusterbased = pd.read_csv('../datasets/training/training_data_ID_numeric_clusterbased.csv')
validation_data = pd.read_csv('../datasets/validation/validation_data_ID_numeric.csv')
test_data = pd.read_csv('../datasets/test/test_data_ID_numeric.csv')
test_data_undersampled = pd.read_csv('../datasets/test/test_data_ID_numeric_undersampled.csv')
validation_data_undersampled = pd.read_csv('../datasets/validation/validation_data_ID_numeric_undersampled_validation.csv')

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

# Function to train and evaluate model
def train_and_evaluate(train_data, label, X_val, y_val, X_tst, y_tst):
    X_train = train_data.drop(columns=['ID', 'loan_status'])
    y_train = train_data['loan_status']
    
    if(label == "original_weight" ):
        svm_model = SVC(kernel='linear', class_weight= {0: 1, 1: 3.5})
    else: 
        svm_model = SVC(kernel='linear')
    
    svm_model.fit(X_train, y_train)
    
    y_pred_validation = svm_model.predict(X_val)
    y_pred_test = svm_model.predict(X_tst)
    
    validation_accuracy = accuracy_score(y_val, y_pred_validation)
    test_accuracy = accuracy_score(y_tst, y_pred_test)
    
    validation_accuracies[label] = validation_accuracy
    test_accuracies[label] = test_accuracy
    
    print(f"{label.capitalize()}:")
    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\n\nValidation Classification Report:\n", classification_report(y_val, y_pred_validation))
    print("Test Classification Report:\n", classification_report(y_tst, y_pred_test))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_pred_validation))
    print("Test Confusion Matrix:\n", confusion_matrix(y_tst, y_pred_test))
    conf_matrix = confusion_matrix(y_tst, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Train individual SVM models
train_and_evaluate(train_data, 'original', X_validation, y_validation, X_test, y_test)
train_and_evaluate(train_data_smote, 'smote', X_validation, y_validation, X_test, y_test)
train_and_evaluate(train_data_adasyn, 'adasyn', X_validation, y_validation, X_test, y_test)
train_and_evaluate(train_data_nearmiss, 'nearmiss', X_validation, y_validation, X_test, y_test)
train_and_evaluate(train_data_clusterbased, 'clusterbased', X_validation, y_validation, X_test, y_test)
train_and_evaluate(train_data_weights_assigned, 'original_weight', X_validation, y_validation, X_test, y_test)

# Train individual SVM models on undersampled data
train_and_evaluate(train_data, 'original_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)
train_and_evaluate(train_data_smote, 'smote_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)
train_and_evaluate(train_data_adasyn, 'adasyn_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)
train_and_evaluate(train_data_nearmiss, 'nearmiss_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)
train_and_evaluate(train_data_clusterbased, 'clusterbased_undersampled', X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)

# Train individual SVM models on undersampled data
train_and_evaluate(test_data_undersampled, 'undersampled_test', X_test_undersampled, y_test_undersampled, X_test, y_test)
train_and_evaluate(validation_data_undersampled, 'undersampled_validation', X_validation_undersampled, y_validation_undersampled, X_test, y_test)

# Ensemble method using VotingClassifier
def train_and_evaluate_ensemble(X_val, y_val, X_tst, y_tst):
    X_train = train_data.drop(columns=['ID', 'loan_status'])
    y_train = train_data['loan_status']
    
    svm_original = SVC(kernel='linear', probability=True)
    svm_smote = SVC(kernel='linear', probability=True)
    svm_adasyn = SVC(kernel='linear', probability=True)
    svm_nearmiss = SVC(kernel='linear', probability=True)
    svm_clusterbased = SVC(kernel='linear', probability=True)
    svm_weights_assigned = SVC(kernel='linear', probability=True)
    
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
    
    validation_accuracy = accuracy_score(y_val, y_pred_validation)
    test_accuracy = accuracy_score(y_tst, y_pred_test)
    
    print("Ensemble Model:")
    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\n\nValidation Classification Report:\n", classification_report(y_val, y_pred_validation))
    print("Test Classification Report:\n", classification_report(y_tst, y_pred_test))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_val, y_pred_validation))
    print("Test Confusion Matrix:\n", confusion_matrix(y_tst, y_pred_test))
    conf_matrix = confusion_matrix(y_tst, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Train ensemble model on original data
train_and_evaluate_ensemble(X_validation, y_validation, X_test, y_test)

# Train ensemble model on undersampled data
train_and_evaluate_ensemble(X_validation_undersampled, y_validation_undersampled, X_test_undersampled, y_test_undersampled)