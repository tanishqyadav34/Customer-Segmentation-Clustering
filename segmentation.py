import pandas as pd

# 1. Load the dataset
try:
    df = pd.read_csv('Mall_Customers.csv')
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'Mall_Customers.csv' was not found. Please check your folder.")
    exit()

# 2. Rename columns
df.columns = ["customer_id", "gender", "age", "annual_income", "spending_score"]

# 3. Preview
print("\n--- FIRST 5 ROWS ---")
print(df.head())
import matplotlib.pyplot as plt
import seaborn as sns

# --- HOUR 2: VISUALIZATION & PREPROCESSING ---

# 1. Select the features we want to cluster
# We only need Annual Income and Spending Score for this project
X = df.iloc[:, [3, 4]].values

# 2. Visualize the raw data
# This creates a simple scatter plot of Income vs Spending
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, c='blue', label='Customer')
plt.title('Customer Groups (Raw Data)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
from sklearn.cluster import KMeans

# --- HOUR 3: THE ELBOW METHOD ---

# 1. Calculate WCSS for 1 to 10 clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# 2. Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# --- HOUR 4: FINAL CLUSTERING ---

# 1. Train the K-Means model with K=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 2. Visualize the Clusters (The "Money Shot")
plt.figure(figsize=(10, 6))

# Plot each cluster with a different color
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=50, c='magenta', label='Cluster 5')

# Plot the Centroids (The center of each group)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()