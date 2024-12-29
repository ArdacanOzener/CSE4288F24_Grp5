#train naive bayes model
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load training datasets
train_datasets = {
    'original': pd.read_csv('datasets/training/training_data_ID_numeric.csv'),
    'smote': pd.read_csv('datasets/training/training_data_ID_numeric_smote.csv'),
    'adasyn': pd.read_csv('datasets/training/training_data_ID_numeric_adasyn.csv'),
    'nearmiss': pd.read_csv('datasets/training/training_data_ID_numeric_nearmiss.csv'),
    'clusterbased': pd.read_csv('datasets/training/training_data_ID_numeric_clusterbased.csv')
}

# Load validation datasets
validation_datasets = {
    'original': pd.read_csv('datasets/validation/validation_data_ID_numeric.csv'),
    'undersampled': pd.read_csv('datasets/validation/validation_data_ID_undersampled_numeric.csv')
}

# Load test datasets
test_datasets = {
    'original': pd.read_csv('datasets/test/test_data_ID_numeric.csv'),
    'undersampled': pd.read_csv('datasets/test/test_data_ID_undersampled_numeric.csv')
}

# Initialize dictionary to store results for each validation dataset
results = {}
for val_name in validation_datasets.keys():
    results[val_name] = {
        'original': {'validation_accuracies': [], 'models': [], 'validation_metrics': [], 'test_metrics': {}},
        'smote': {'validation_accuracies': [], 'models': [], 'validation_metrics': [], 'test_metrics': {}},
        'adasyn': {'validation_accuracies': [], 'models': [], 'validation_metrics': [], 'test_metrics': {}},
        'nearmiss': {'validation_accuracies': [], 'models': [], 'validation_metrics': [], 'test_metrics': {}},
        'clusterbased': {'validation_accuracies': [], 'models': [], 'validation_metrics': [], 'test_metrics': {}}
    }

# Define var_smoothing values to test
var_smoothing_values = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

def train_and_evaluate_model(training_data, validation_data, label, val_name):
    X_train = training_data.drop(['ID', 'loan_status'], axis=1)
    y_train = training_data['loan_status']
    
    X_validation = validation_data.drop(['ID', 'loan_status'], axis=1)
    y_validation = validation_data['loan_status']
    
    for var_smoothing in var_smoothing_values:
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(X_train, y_train)
        
        # Validation predictions and metrics
        y_pred_validation = model.predict(X_validation)
        validation_accuracy = accuracy_score(y_validation, y_pred_validation)
        validation_precision = precision_score(y_validation, y_pred_validation)
        validation_recall = recall_score(y_validation, y_pred_validation)
        validation_f1 = f1_score(y_validation, y_pred_validation)
        validation_conf_matrix = confusion_matrix(y_validation, y_pred_validation)
        
        # Store results
        results[val_name][label]['validation_accuracies'].append(validation_accuracy)
        results[val_name][label]['models'].append(model)
        results[val_name][label]['validation_metrics'].append({
            'var_smoothing': var_smoothing,
            'accuracy': validation_accuracy,
            'precision': validation_precision,
            'recall': validation_recall,
            'f1': validation_f1,
            'confusion_matrix': validation_conf_matrix
        })

# Train and evaluate models for each combination of training and validation datasets
for val_name, val_data in validation_datasets.items():
    for train_name, train_data in train_datasets.items():
        train_and_evaluate_model(train_data, val_data, train_name, val_name)
    
    # Create visualization for each validation dataset
    plt.figure(figsize=(15, 10))
    bar_width = 0.15
    index = np.arange(len(var_smoothing_values))
    
    for i, dataset_name in enumerate(train_datasets.keys()):
        accuracies = results[val_name][dataset_name]['validation_accuracies']
        plt.bar(index + i * bar_width, accuracies, bar_width, 
                label=dataset_name.capitalize(), alpha=0.8)
    
    plt.xlabel('var_smoothing Values')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Validation Accuracy Comparison - {val_name.capitalize()} Validation Set')
    plt.xticks(index + bar_width * 2, [f'{val:.0e}' for val in var_smoothing_values], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Create a heatmap of validation accuracies
    accuracy_matrix = np.zeros((len(train_datasets.keys()), len(var_smoothing_values)))
    for i, dataset_name in enumerate(train_datasets.keys()):
        accuracy_matrix[i] = results[val_name][dataset_name]['validation_accuracies']
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(accuracy_matrix, 
                xticklabels=[f'{val:.0e}' for val in var_smoothing_values],
                yticklabels=list(train_datasets.keys()),
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd')
    plt.xlabel('var_smoothing Values')
    plt.ylabel('Training Set')
    plt.title(f'Validation Accuracy Heatmap - {val_name.capitalize()} Validation Set')
    plt.tight_layout()
    plt.show()

# Test each training dataset's best model
for val_name in validation_datasets.keys():
    print(f"\nResults for {val_name.upper()} validation dataset:")
    print("-" * 50)
    print("\nTested var_smoothing values:", ", ".join([f"{val:.0e}" for val in var_smoothing_values]))
    print("-" * 50)
    
    for train_name in train_datasets.keys():
        # Find best model for this training dataset
        best_idx = np.argmax(results[val_name][train_name]['validation_accuracies'])
        best_model = results[val_name][train_name]['models'][best_idx]
        best_var_smoothing = var_smoothing_values[best_idx]
        best_validation_metrics = results[val_name][train_name]['validation_metrics'][best_idx]
        
        print(f"\nBest model for {train_name.upper()} training set:")
        print(f"Best var_smoothing: {best_var_smoothing:.0e}")
        
        # Print all validation accuracies with their var_smoothing values
        print("\nAll validation accuracies:")
        for idx, var_smooth in enumerate(var_smoothing_values):
            acc = results[val_name][train_name]['validation_accuracies'][idx]
            print(f"  var_smoothing {var_smooth:.0e}: {acc:.4f}")
        
        print(f"\nBest Validation Metrics (var_smoothing = {best_var_smoothing:.0e}):")
        print(f"  Accuracy:  {best_validation_metrics['accuracy']:.4f}")
        print(f"  Precision: {best_validation_metrics['precision']:.4f}")
        print(f"  Recall:    {best_validation_metrics['recall']:.4f}")
        print(f"  F1-Score:  {best_validation_metrics['f1']:.4f}")
        
        # Test on all test datasets
        for test_name, test_data in test_datasets.items():
            X_test = test_data.drop(['ID', 'loan_status'], axis=1)
            y_test = test_data['loan_status']
            
            y_pred_test = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test)
            test_recall = recall_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test)
            test_conf_matrix = confusion_matrix(y_test, y_pred_test)
            
            results[val_name][train_name]['test_metrics'][test_name] = {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'confusion_matrix': test_conf_matrix,
                'var_smoothing': best_var_smoothing
            }
            
            print(f"\nTest Results on {test_name.upper()} test set (var_smoothing = {best_var_smoothing:.0e}):")
            print(f"  Accuracy:  {test_accuracy:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall:    {test_recall:.4f}")
            print(f"  F1-Score:  {test_f1:.4f}")
            print("\nConfusion Matrix:")
            print(test_conf_matrix)

# Create comparison plots for test results
for val_name in validation_datasets.keys():
    for test_name in test_datasets.keys():
        # Prepare data for plotting
        train_names = list(train_datasets.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = {metric: [] for metric in metrics}
        var_smoothing_used = []
        
        for train_name in train_names:
            for metric in metrics:
                metric_values[metric].append(
                    results[val_name][train_name]['test_metrics'][test_name][metric]
                )
            var_smoothing_used.append(
                results[val_name][train_name]['test_metrics'][test_name]['var_smoothing']
            )
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(train_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, metric_values[metric], width, label=metric.capitalize())
        
        plt.xlabel('Training Dataset')
        plt.ylabel('Score')
        title = f'Test Metrics on {test_name.upper()} Test Set\n(Models selected using {val_name.upper()} validation set)\n'
        title += 'var_smoothing values: ' + ', '.join([f'{v:.0e}' for v in var_smoothing_used])
        plt.title(title)
        plt.xticks(x + width * 1.5, [name.capitalize() for name in train_names], rotation=45)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()