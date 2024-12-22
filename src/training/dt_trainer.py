import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import numpy as np

# Load the dataset
train_data = pd.read_csv('../../datasets/training/training_data_ID_DTB.csv')  # Replace with the path to your training data
validation_data = pd.read_csv('../../datasets/validation/validation_data_ID_DTB.csv')  # Replace with the path to your validation data

# Split the data into features and target
X_train = train_data.drop(columns=['loan_status', "ID"], axis=1)  # Replace 'target' with your target column
y_train = train_data['loan_status']
X_valid = validation_data.drop(columns=['loan_status', "ID"], axis=1)  # Replace 'target' with your target column
y_valid = validation_data['loan_status']

# Hyperparameters to tune
max_depths = [3, 5, 10, None]  # Example of different max_depth values for Decision Trees
min_samples_splits = [2, 5, 10]  # Example of different min_samples_split values

# Store models and their corresponding losses
models = []
losses = []

# Iterate through combinations of hyperparameters
for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        # Train a Decision Tree model
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        model.fit(X_train, y_train)
        
        # Make predictions on the validation set
        y_pred = model.predict(X_valid)
        
        # Compute log loss (or accuracy as alternative)
        loss = log_loss(y_valid, model.predict_proba(X_valid))  # Using log loss here, you can use other metrics
        
        # Store model and its corresponding loss
        models.append((max_depth, min_samples_split, model))
        losses.append(loss)

# Visualize the losses
loss_matrix = np.array(losses).reshape(len(max_depths), len(min_samples_splits))

plt.figure(figsize=(10, 6))
plt.imshow(loss_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Model Validation Loss")
plt.xlabel("Min Samples Split")
plt.ylabel("Max Depth")
plt.xticks(np.arange(len(min_samples_splits)), min_samples_splits)
plt.yticks(np.arange(len(max_depths)), max_depths)
plt.colorbar()

# Annotating the heatmap with loss values
for i in range(len(max_depths)):
    for j in range(len(min_samples_splits)):
        plt.text(j, i, f'{loss_matrix[i, j]:.4f}', ha='center', va='center', color='red')

plt.savefig("decision_tree_loss.png")

# Best model based on the lowest log loss
best_model_idx = np.argmin(losses)
best_model = models[best_model_idx]
print(f"Best model parameters: Max Depth = {best_model[0]}, Min Samples Split = {best_model[1]}")
