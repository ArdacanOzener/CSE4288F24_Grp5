import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Load the decision tree model from a pickle file
def load_model(pickle_file):
    with open(pickle_file, 'rb') as file:
        model = pickle.load(file)
    return model

# Load and preprocess the dataset
def preprocess_dataset(csv_file, model_feature_names):
    # Load dataset
    data = pd.read_csv(csv_file)


    X = pd.get_dummies(data.drop(columns=['loan_status', "ID"], axis=1))  # Replace 'target' with your target column
    y = data['loan_status']

    
    # Ensure all columns used during model training are present
    for col in model_feature_names:
        if col not in X.columns:
            X[col] = 0  # Add missing columns with default value
    
    # Remove any columns that are not in model_feature_names
    X = X[model_feature_names]
    
    return X, y

# Plot confusion matrix with metrics side by side
def plot_confusion_matrix_with_metrics(y_true, y_pred, file_name):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall  # Same as recall for binary classification
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Create a figure with two subplots (one for confusion matrix, one for metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

    # Confusion Matrix Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(file_name)

    # Metrics Text
    metrics_text = f"Accuracy: {accuracy:.2f}\n" \
                   f"Precision: {precision:.2f}\n" \
                   f"Recall (Sensitivity): {sensitivity:.2f}\n" \
                   f"Specificity: {specificity:.2f}\n" \
                   f"F1 Score: {f1:.2f}"

    ax2.text(0.1, 0.8, metrics_text, fontsize=12, verticalalignment='top')
    ax2.axis('off')  # Turn off axes for the metrics plot
    ax2.set_title('Metrics')

    plt.tight_layout()
    plt.savefig(f"{file_name}confusion_matrix_with_metrics.png")


# Main function
def main():
    for val in ["DTB", "frequency", "range"]:
        for meth in ["adasyn", "smote", "nearmiss", "clusterbased", "unweighted", "weighted"]:
        #for meth in ["unweighted", "weighted"]:
            pickle_file = f"../../models/DecisionTree/{val}_{meth}/{val}_{meth}-tree-model.pickle"  # Replace with your pickle file path
            csv_file = f"../../datasets/test/test_data_ID_undersampled_{val}.csv"  # Replace with your CSV file path


            # Load model
            model = load_model(pickle_file)

            # Get feature names from the trained model
            if hasattr(model, 'feature_names_in_'):
                model_feature_names = model.feature_names_in_.tolist()
            else:
                raise ValueError("The model does not have feature names. Ensure the model was trained with named features.")

            # Preprocess dataset
            X, y = preprocess_dataset(csv_file, model_feature_names)

            # Predict on the dataset
            y_pred = model.predict(X)

            # Plot confusion matrix with metrics
            plot_confusion_matrix_with_metrics(y, y_pred, f"{val}_{meth}_forest")

if __name__ == "__main__":
    main()
