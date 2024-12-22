import pandas as pd
import json
import ast
import os

def convert_to_categorical(dataset_path, json_path, output_path):
    # Load the CSV dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")
    
    df = pd.read_csv(dataset_path)

    # Load the JSON configuration
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")
    
    with open(json_path, 'r') as json_file:
        column_ranges = json.load(json_file)

    # Process each specified column
    for column, ranges in column_ranges.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataset. Skipping.")
            continue

        # Create a mapping of ranges to category labels
        range_mapping = {}
        for i, range_str in enumerate(ranges):
            # Parse the range string, e.g., "[20.00, 32.40]"
            range_tuple = ast.literal_eval(range_str.replace('[', '(').replace(']', ')'))
            range_mapping[range_tuple] = str(range_tuple)

        # Map numeric values to categories
        def map_to_category(value):
            for range_tuple, category in range_mapping.items():
                if range_tuple[0] <= value <= range_tuple[1]:
                    return category
            return "Unknown"

        df[column] = df[column].apply(map_to_category)

    # Save the modified dataset
    df.to_csv(output_path, index=False)
    print(f"Converted dataset saved to '{output_path}'.")

# Example usage
for j in ["DTB", "frequency", "range"]:
    for i in ["adasyn", "smote", "clusterbased", "nearmiss"]:
        convert_to_categorical(
            dataset_path=f"../../tmp_datasets/training_data_ID_{i}.csv",
            json_path='../../config_files/DT_ranges.json',
            output_path=f'../../datasets/training/training_data_ID_{j}_{i}.csv'
        )
