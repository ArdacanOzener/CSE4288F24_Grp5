import json
import pandas as pd
import numpy as np

dataframe_path = 'datasets/loan_data.csv'
column_min_max_values_path = 'column_min_max_values.json'




def save_min_max_values(df, file_path):
    min_max_values = {}
    #only convert numerical columns
    for column in df.columns:
        if df[column].dtype in [np.float64, np.int64]:
            min_max_values[column] = {
                'min': float(df[column].min()),
                'max': float(df[column].max())
            }
    
    #save to json
    with open(file_path, 'w') as file:
        json.dump(min_max_values, file)
    
    return min_max_values

def denormalize_column(df, column, min_max_values):
    min_value = min_max_values[column]['min']
    max_value = min_max_values[column]['max']
    return df[column] * (max_value - min_value) + min_value

def denormalize_dataframe(df, min_max_values_json_path):
    min_max_values = json.load(open(min_max_values_json_path))
    for column in df.columns:
        if column in min_max_values:
            df[column] = denormalize_column(df, column, min_max_values)
    return df


df = pd.read_csv(dataframe_path)
df = save_min_max_values(df, column_min_max_values_path)

