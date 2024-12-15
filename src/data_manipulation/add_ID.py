import pandas as pd

def add_id_to_csv(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    print(len(df))
    
    # Add a new column "ID" with unique identifiers starting from 1
    df.insert(0, "ID", range(40500, len(df) + 40500))

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"ID column added and saved to {output_file}")

# Example usage
input_file = "../../test_data.csv"  # Replace with the path to your input CSV file
output_file = "../../datasets/test_data_ID.csv"  # Replace with the path to your output CSV file
add_id_to_csv(input_file, output_file)
