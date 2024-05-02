import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and split the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def calculate_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid) ** 2))

def k_means(X, k=3, max_iterations=10, distance_threshold=6):
    # Initialize centroids
    random_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[random_indices, :]

    for iteration in range(max_iterations):
        # Assign clusters
        clusters = [[] for _ in range(k)]
        for idx, point in enumerate(X):
            distances = [calculate_euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(idx)

        # Update centroids
        new_centroids = np.array([np.mean(X[cluster], axis=0) for cluster in clusters])

        # Check for convergence
        shifts = [calculate_euclidean_distance(centroids[i], new_centroids[i]) for i in range(k)]
        if sum(shifts) <= distance_threshold:
            break
        centroids = new_centroids

    # Calculate WCSS
    wcss = sum(np.sum((X[cluster] - centroids[i]) ** 2) for i, cluster in enumerate(clusters))
    # enumerate gets both the index and the value of each item in the iterable
    return wcss, centroids

# Plotting WCSS to find the best k
k_values = range(1, 6)
WCSS = [k_means(X_train_scaled, k)[0] for k in k_values]

plt.plot(k_values, WCSS)
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method For Optimal k')
plt.show()
