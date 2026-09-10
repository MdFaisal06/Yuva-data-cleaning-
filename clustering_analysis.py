import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Dataset Generation (Customer Income and Spending Behavior)
np.random.seed(42)
income = np.concatenate([
    np.random.normal(25, 5, 40), np.random.normal(55, 8, 50),
    np.random.normal(85, 8, 40), np.random.normal(25, 5, 35),
    np.random.normal(88, 7, 35)
])
spending = np.concatenate([
    np.random.normal(20, 6, 40), np.random.normal(50, 7, 50),
    np.random.normal(82, 6, 40), np.random.normal(80, 7, 35),
    np.random.normal(18, 6, 35)
])

df = pd.DataFrame({
    'CustomerID': range(1, len(income) + 1),
    'Annual_Income_k$': income.round(1),
    'Spending_Score': spending.round(1)
})

X = df[['Annual_Income_k$', 'Spending_Score']].values

# 2. Elbow Method to Determine Optimal k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', color='#182b49')
plt.title("Elbow Method for Optimal k", fontsize=11, fontweight='bold')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('elbow_method.png')
plt.close()

# 3. Fitting K-Means with Optimal k=5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# 4. Visualizing Customer Segments
plt.figure(figsize=(7, 5))
colors = ['#2b5c8f', '#7570b3', '#1b9e77', '#e7298a', '#d95f02']
cluster_labels = ['Sensible', 'Average', 'Target (High Value)', 'Careless', 'Careful']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual_Income_k$'], cluster_data['Spending_Score'],
                s=40, color=colors[i], label=f"Cluster {i+1}: {cluster_labels[i]}")

# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=120, color='black', marker='X', label='Centroids')

plt.title("Customer Segmentation via K-Means (k=5)", fontsize=11, fontweight='bold')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('customer_clusters.png')
plt.close()

# 5. Export Segmented Data
df.to_csv('customer_clusters.csv', index=False)
print("Week 3: K-Means Clustering Analysis executed successfully.")
