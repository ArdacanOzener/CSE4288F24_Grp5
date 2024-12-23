#train naive bayes model
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('datasets/training/training_data_ID_numeric.csv')
train_data_smote = pd.read_csv('datasets/training/training_data_ID_numeric_smote.csv')
train_data_adasyn = pd.read_csv('datasets/training/training_data_ID_numeric_adasyn.csv')
train_data_nearmiss = pd.read_csv('datasets/training/training_data_ID_numeric_nearmiss.csv')
train_data_clusterbased = pd.read_csv('datasets/training/training_data_ID_numeric_clusterbased.csv')

validation_data = pd.read_csv('datasets/validation/validation_data_ID_numeric.csv')
test_data = pd.read_csv('datasets/test/test_data_ID_numeric.csv')

# Initialize dictionaries to store accuracies
validation_accuracies = {}
test_accuracies = {}

def train_and_evaluate_model(training_data, label):
    X_train = training_data.drop(['ID', 'loan_status'], axis=1)
    y_train = training_data['loan_status']
    

    model = GaussianNB()

    model.fit(X_train, y_train)
    #Test the model
   
    X_test = test_data.drop(['ID', 'loan_status'], axis=1)
    y_test = test_data['loan_status']

    X_validation = validation_data.drop(['ID', 'loan_status'], axis=1)
    y_validation = validation_data['loan_status']

    y_pred_test = model.predict(X_test)
    y_pred_validation = model.predict(X_validation)

    validation_accuracy = accuracy_score(y_validation, y_pred_validation)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    validation_accuracies[label] = validation_accuracy
    test_accuracies[label] = test_accuracy
    
    print(f"{label.capitalize()}:")
    print("Validation Accuracy:", validation_accuracy)
    print("Test Accuracy:", test_accuracy)
    print("\n\nValidation Classification Report:\n", classification_report(y_validation, y_pred_validation))
    print("Test Classification Report:\n", classification_report(y_test, y_pred_test))
    print("Validation Confusion Matrix:\n", confusion_matrix(y_validation, y_pred_validation))
    print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {label.capitalize()} Dataset')
    plt.show()

train_and_evaluate_model(train_data, "original")
train_and_evaluate_model(train_data_smote, "smote")
train_and_evaluate_model(train_data_adasyn, "adasyn")
train_and_evaluate_model(train_data_nearmiss, "nearmiss")
train_and_evaluate_model(train_data_clusterbased, "clusterbased")