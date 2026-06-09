import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\Navigate lab\Datasets\AB_NYC_2019.csv")
print(df.head())

# prepare data
print(df['price'].describe())
print(df.shape)

# Calculate percentiles
min_threshold = df['price'].quantile(0.01)
max_threshold = df['price'].quantile(0.99)
print(f"1st Percentile (Min Threshold): {min_threshold}")
print(f"99th Percentile (Max Threshold): {max_threshold}")

# Filter out outliers
filtered_df = df[(df['price'] >= min_threshold) & (df['price'] <= max_threshold)]
print(filtered_df.shape)
print(filtered_df['price'].describe())