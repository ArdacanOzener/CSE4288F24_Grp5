import pandas as pd

# Create two DataFrames with overlapping but not identical columns
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"B": [5, 6], "C": [7, 8]})

# Align using a left join
aligned_df1, aligned_df2 = df1.align(df2, join='outer', axis=1, fill_value=0)

# Print the results
print("Aligned df1:")
print(aligned_df1)
print("\nAligned df2:")
print(aligned_df2)
