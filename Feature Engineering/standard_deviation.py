import pandas as pd

# Load dataset
df = pd.read_csv(r"D:\Navigate lab\Datasets\bhp.csv")

# prepare data
print(df.head())
print(df['price_per_sqft'].describe())
print(df.shape)

#calculate using percentiles
min_threshold = df['price_per_sqft'].quantile(0.001)
max_threshold = df['price_per_sqft'].quantile(0.999)
print(f"1st Percentile (Min Threshold): {min_threshold}")
print(f"999th Percentile (Max Threshold): {max_threshold}")

# Filter out outliers
filtered_df = df[(df['price_per_sqft'] >= min_threshold) & (df['price_per_sqft'] <= max_threshold)]
print(filtered_df.shape)

# Calculate standard deviation
std_dev = filtered_df['price_per_sqft'].std()
print(f"Standard Deviation of price_per_sqft: {std_dev}")
min_limit = filtered_df['price_per_sqft'].mean() - 4 * std_dev
max_limit = filtered_df['price_per_sqft'].mean() + 4 * std_dev
print(f"Lower Limit: {min_limit}")
print(f"Upper Limit: {max_limit}")
filtered_df2 = filtered_df[(filtered_df['price_per_sqft'] >= min_limit) & (filtered_df['price_per_sqft'] <= max_limit)]
print(filtered_df2.shape)