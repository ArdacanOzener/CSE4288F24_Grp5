import json

# Load the previous JSON result
with open("../../config_files/bin_analysis.json", "r") as file:
    feature_results = json.load(file)

# Load the min and max values for each feature
with open("../../config_files/column_min_max_values.json", "r") as file:
    min_max_values = json.load(file)

ranges = {}

# Iterate over each feature's analysis data
for feature_data in feature_results:
    for feature, bin_results in feature_data.items():
        min_entropy = float('inf')
        selected_thresholds = []

        # Iterate through different max_leaf_nodes configurations
        for bin_result in bin_results:
            for max_leaf_nodes, results in bin_result.items():
                # Extract relevant data
                thresholds = results[0]["Thresholds"]
                average_leaf_entropy = results[1]["Average Leaf Entropy"]

                # Find the minimum Average Leaf Entropy
                if average_leaf_entropy < min_entropy:
                    min_entropy = average_leaf_entropy
                    selected_thresholds = thresholds

        # Convert thresholds to floats and sort
        selected_thresholds = list(map(float, selected_thresholds))
        selected_thresholds.sort()

        # Get the min and max values for the feature
        min_value = min_max_values[feature]["min"]
        max_value = min_max_values[feature]["max"]

        # Create the ranges, ensuring the min and max values are used
        feature_ranges = []

        # Add the range for min to the first threshold
        feature_ranges.append(f"[{min_value}, {selected_thresholds[0]}]")

        # Add the intermediate ranges based on the thresholds
        for i in range(1, len(selected_thresholds)):
            feature_ranges.append(f"[{selected_thresholds[i-1]}, {selected_thresholds[i]}]")

        # Add the range for the last threshold to max value
        feature_ranges.append(f"[{selected_thresholds[-1]}, {max_value}]")

        # Save the ranges for the feature
        ranges[feature] = feature_ranges

# Save the ranges to a new JSON file
output_file_path = "DT_ranges.json"
with open(output_file_path, "w") as output_file:
    json.dump(ranges, output_file, indent=4)

print("Feature ranges with min and max values saved to", output_file_path)
