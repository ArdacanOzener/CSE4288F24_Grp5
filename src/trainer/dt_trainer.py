import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import numpy as np
import seaborn as sns
import json
import time
import pickle


for val in ["DTB"]:#, "frequency", "range"]:
    if True:
    #for meth in ["adasyn", "smote", "nearmiss", "clusterbased"]:

        # Load the dataset
        train_path = f'../../datasets/training/training_data_ID_{val}_{meth}.csv'
        train_data = pd.read_csv(train_path)  # Replace with the path to your training data
        validation_data = pd.read_csv(f'../../datasets/validation/validation_data_ID_{val}.csv') # Replace with the path to your validation data
        


        # Split the data into features and target
        X_train = pd.get_dummies(train_data.drop(columns=['loan_status', "ID"], axis=1))  # Replace 'target' with your target column
        y_train = train_data['loan_status']
        X_valid = pd.get_dummies(validation_data.drop(columns=['loan_status', "ID"], axis=1))  # Replace 'target' with your target column
        y_valid = validation_data['loan_status']

        X_train, X_valid = X_train.align(X_valid, join='outer', axis=1, fill_value=0)




        # Hyperparameters to tune
        max_depths = [30, 50, 100, 300]  # Example of different max_depth values for Decision Trees
        min_samples_splits = [2, 5, 10]  # Example of different min_samples_split values

        # Store models and their corresponding losses
        models = []
        losses = []
        configurations = []

        # Iterate through combinations of hyperparameters
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                # Train a Decision Tree model
                class_weight={0: 1, 1: 3.5}
                class_weight=None
                start_time = time.time()
                model = RandomForestClassifier(class_weight=class_weight, n_estimators=max_depth, min_samples_split=min_samples_split)
                model.fit(X_train, y_train)
                end_time = time.time()
                
                # Make predictions on the validation set
                y_pred = model.predict(X_valid)
                
                # Compute log loss (or accuracy as alternative)
                loss = log_loss(y_valid, model.predict_proba(X_valid))  # Using log loss here, you can use other metrics

                # Store model, its hyperparameters, and corresponding loss
                models.append((max_depth, min_samples_split, model))
                losses.append(loss)
                configurations.append({
                    'training dataset': train_path,
                    'n_estimators': max_depth,
                    'min_samples_split': min_samples_split,
                    'training time': end_time - start_time,
                    'loss': loss
                })
                


        # Visualize the losses
        loss_matrix = np.array(losses).reshape(len(max_depths), len(min_samples_splits))

        plt.figure(figsize=(10, 6))
        plt.imshow(loss_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{val}_{meth} Model Validation Loss")
        plt.xlabel("Min Samples Split")
        plt.ylabel("Number of Estimators")
        plt.xticks(np.arange(len(min_samples_splits)), min_samples_splits)
        plt.yticks(np.arange(len(max_depths)), max_depths)
        plt.colorbar()

        # Annotating the heatmap with loss values
        for i in range(len(max_depths)):
            for j in range(len(min_samples_splits)):
                plt.text(j, i, f'{loss_matrix[i, j]:.4f}', ha='center', va='center', color='red')

        plt.savefig(f"./fig/{val}_{meth}_forest_loss.png")

        # Best model based on the lowest log loss
        best_model_idx = np.argmin(losses)
        best_model = models[best_model_idx]
        best_config = configurations[best_model_idx]

        print(f"Best model parameters: Max Depth = {best_model[0]}, Min Samples Split = {best_model[1]}")





        # save object to pickle file
        with open(f"{val}_{meth}-forest-model.pickle", "wb") as fout:
            pickle.dump(best_model[2], fout)

        # Save the hyperparameters and configuration to a JSON file
        config_filename = f"{val}_{meth}_forest_config.json"
        with open(config_filename, 'w') as f:
            json.dump(best_config, f, indent=4)

        print(f"Model and configuration saved to {config_filename}")
