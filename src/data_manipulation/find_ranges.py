import pandas as pd
import numpy as np
import json

def equal_frequency_binning(input_file, output_file, columns, bins):
    """
    Perform equal-frequency binning on specified numeric columns of a dataset and save bin ranges to a JSON file.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        columns (list): List of numeric columns to bin.
        bins (int): Number of bins for equal-frequency binning.
        json_file (str): Path to save bin ranges as a JSON file.
    """

    json_file = output_file
    # Load dataset
    df = pd.read_csv(input_file)

    bin_ranges = {}

    # Apply equal-frequency binning to specified columns
    for col in columns:
        if col in df.columns:
            try:
                # Create bins with range labels
                bin_labels = []
                bin_edges = pd.qcut(df[col], q=bins, retbins=True, duplicates='drop')[1]
                for i in range(len(bin_edges) - 1):
                    bin_labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]")

                # Assign range labels to binned column
                df[col] = pd.qcut(df[col], q=bins, labels=bin_labels, duplicates='drop')

                # Store bin ranges in dictionary
                bin_ranges[col] = bin_labels
            except ValueError as e:
                print(f"Error binning column '{col}': {e}")
        else:
            print(f"Column '{col}' not found in the dataset.")

    # Move 'loan_status' column to the end if it exists
    if 'loan_status' in df.columns:
        loan_status = df.pop('loan_status')
        df['loan_status'] = loan_status


    # Save bin ranges to JSON file
    with open(json_file, 'w') as f:
        json.dump(bin_ranges, f, indent=4)
    print(f"Bin ranges saved to {json_file}")

def equal_range_binning(input_file, output_file, columns, bins):
    """
    Perform equal-range binning on specified numeric columns of a dataset and save bin ranges to a JSON file.

    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
        columns (list): List of numeric columns to bin.
        bins (int): Number of bins for equal-range binning.
        json_file (str): Path to save bin ranges as a JSON file.
    """

    json_file = output_file
    # Load dataset
    df = pd.read_csv(input_file)

    bin_ranges = {}

    # Apply equal-range binning to specified columns
    for col in columns:
        if col in df.columns:
            try:
                # Create bins with range labels
                bin_labels = []
                bin_edges = np.linspace(df[col].min(), df[col].max(), bins + 1)
                for i in range(len(bin_edges) - 1):
                    bin_labels.append(f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}]")

                # Assign range labels to binned column
                df[col] = pd.cut(df[col], bins=bin_edges, labels=bin_labels, include_lowest=True)

                # Store bin ranges in dictionary
                bin_ranges[col] = bin_labels
            except ValueError as e:
                print(f"Error binning column '{col}': {e}")
        else:
            print(f"Column '{col}' not found in the dataset.")

    # Move 'loan_status' column to the end if it exists
    if 'loan_status' in df.columns:
        loan_status = df.pop('loan_status')
        df['loan_status'] = loan_status


    # Save bin ranges to JSON file
    with open(json_file, 'w') as f:
        json.dump(bin_ranges, f, indent=4)
    print(f"Bin ranges saved to {json_file}")


# Example usage

input_csv = "../../datasets/loan_data.csv"  # Path to input CSV
output_json = "../../config_files/"  # Path to save output CSV
numeric_columns = ["person_age", "person_income", 
           "person_emp_exp", "loan_amnt", 
           "loan_int_rate", "loan_percent_income",
           "cb_person_cred_hist_length", "credit_score"] # Replace with your column names
number_of_bins = 10  # Replace with the desired number of bins

equal_frequency_binning(input_csv, output_json+"frequency_config.json", numeric_columns, number_of_bins)
equal_range_binning(input_csv, output_json+"range_config.json", numeric_columns, number_of_bins)
