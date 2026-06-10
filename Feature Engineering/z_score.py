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

# Calculate mean and standard deviation
mean_price = filtered_df['price_per_sqft'].mean()
std_dev_price = filtered_df['price_per_sqft'].std()
print(f"Mean Price: {mean_price}")
print(f"Standard Deviation of Price: {std_dev_price}")

# Calculate z-scores and filter out outliers
filtered_df['z_score'] = (filtered_df['price_per_sqft'] - mean_price) / std_dev_price
filtered_df = filtered_df[filtered_df['z_score'].abs() <= 4]  # Keep rows with z-scores within 3 standard deviations
print(filtered_df.shape)
