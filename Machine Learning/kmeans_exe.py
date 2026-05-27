from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pandas as pd

# Load dataset
data = pd.read_csv(r'D:\Navigate lab\Datasets\Mall_Customers.csv')
print(data.head())

# Prepare data
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], color='blue')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Data')
plt.show()


# elbow method to find optimal number of clusters
sse = []
for i in range(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(data[['Annual Income (k$)', 'Spending Score (1-100)']])
    sse.append(km.inertia_)

plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors')
plt.title('Elbow Method')
plt.show()

# Fit KMeans with optimal clusters
km = KMeans(n_clusters=5)
km.fit(data[['Annual Income (k$)', 'Spending Score (1-100)']])
y_km = km.predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])
data['Cluster'] = y_km

# Plot clusters
plt.scatter(data[data['Cluster'] == 0]['Annual Income (k$)'], data[data['Cluster'] == 0]['Spending Score (1-100)'], color='red', label='Cluster 1')
plt.scatter(data[data['Cluster'] == 1]['Annual Income (k$)'], data[data['Cluster'] == 1]['Spending Score (1-100)'], color='blue', label='Cluster 2')
plt.scatter(data[data['Cluster'] == 2]['Annual Income (k$)'], data[data['Cluster'] == 2]['Spending Score (1-100)'], color='green', label='Cluster 3')
plt.scatter(data[data['Cluster'] == 3]['Annual Income (k$)'], data[data['Cluster'] == 3]['Spending Score (1-100)'], color='cyan', label='Cluster 4')
plt.scatter(data[data['Cluster'] == 4]['Annual Income (k$)'], data[data['Cluster'] == 4]['Spending Score (1-100)'], color='magenta', label='Cluster 5')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='yellow', marker='X', s=200, label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters')
plt.legend()
plt.show()