## K Means Implementation

## Design Template
#   1. Pick K random data points from the dataset as centroids
#   2. Calculate distance (Euclidean, Manhattan etc) for every data point in the dataset to the K centroids
#   3. Assign every data points to one of the K centroids depending on the shortest distance to them
#   4. Recalculate the centroids as average of data points in each of the K centroids
#   5. Shift the centroid to the new centroids and repeat the steps all over untill convergence (Convergence occur when no centroids doesnt move or move less than the threhold value)
#   6. Return the cluster labels

import numpy as np

class KMeans:
    def __init__(self, K, max_iterations = 500, delta = 1e-3):
        self.K =K
        self.max_iterations = max_iterations
        self.delta = delta
        self.k_centroids = []
        self.clusters =[[] for _ in range(self.K)]
        self._has_converged =False

    def predict(self, X):
        self.X =X
        self.n_samples, self.n_features = X.shape
        
        # pick K random centroids
        k_ids = np.random.choice(self.n_samples, self.K, replace=False)
        self.k_centroids = self.X[k_ids,:]

        cluster_labels =self._perform_clustering(num_iterations =1)
        return cluster_labels
    
    def _perform_clustering(self, num_iterations):
        
        # Assign cluster labels
        self.clusters =self._create_clusters(self.X)

        #Terminating criteria
        if num_iterations >=self.max_iterations or self._has_converged :
            return self._get_cluster_labels(self.clusters)

        # Compute new centroids
        new_centroids= self._compute_centroids(self.clusters)
        # Check for convergence
        self._has_converged =self._check_convergence(new_centroids)
        self.k_centroids =new_centroids

        return self._perform_clustering(num_iterations+1)
    
    def _create_clusters(self, X):
        clusters =[[] for _ in range(self.K)]
        for idx, x in enumerate(X):
            cluster_id =self._assign_clusters(x)
            clusters[cluster_id].append(idx)
        return clusters

    def _assign_clusters(self, x):
        distances = [self._euclidean(x, centroid) for centroid in self.k_centroids]
        return np.argmin(distances)
    
    def _euclidean(self, x1, x2):
        return np.sqrt(np.sum(np.square(x1-x2)))

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] =idx
        return labels

    def _compute_centroids(self, clusters ):
        centroids = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            cluster_mean =np.mean(self.X[cluster], axis=0)
            centroids[idx] = cluster_mean
        return centroids
    
    def _check_convergence(self, new_centroids):
        delta_est = np.array([self._euclidean(x1, x2) for x1, x2 in zip(new_centroids, self.k_centroids)])
        return False if sum(delta_est <= self.delta) < self.K else True


## Testing

if __name__ == "__main__":
    np.random.seed(123)

    from sklearn.datasets import make_blobs
    X, y = make_blobs(centers= 5, n_samples= 400, n_features=2, shuffle=True, random_state= 123)
    print(X.shape)
    
    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iterations=200)
    y_pred =k.predict(X)
    print(y_pred)

