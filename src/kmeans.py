"""K-means clustering, implemented from scratch.

This follows the exact algorithm from my presentation
(`presentation/Presentation_beta3.pptx`, Fig. 1):

    Start -> Choose Nr. of clusters (K)
          -> Randomly set centroids
          -> Distance of each object to the centroids
          -> Group on minimum distance
          -> Recompute centroids; repeat until no object changes group

The point of writing it by hand is to *understand* it — for real work,
use `sklearn.cluster.KMeans` (rule of thumb: write it once by hand,
then use the library forever).
"""

import numpy as np


class KMeansScratch:
    """Plain K-means with the same core attributes as scikit-learn's version.

    Parameters
    ----------
    n_clusters : int
        K — the number of clusters to form.
    max_iter : int
        Hard stop even if the assignments never settle.
    n_init : int
        How many random restarts to try; the best run (lowest inertia) wins.
        This is the standard defence against a bad random initialisation.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_clusters=3, max_iter=300, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def _single_run(self, X, rng):
        # Step 1: random initial "seeds" — K distinct points from the data
        seed_idx = rng.choice(len(X), size=self.n_clusters, replace=False)
        centers = X[seed_idx].copy()

        labels = np.full(len(X), -1)
        for n_iter in range(1, self.max_iter + 1):
            # Step 2: squared Euclidean distance from every point to every centroid
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

            # Step 3: assign each point to its nearest centroid
            new_labels = distances.argmin(axis=1)

            # Convergence check: nobody changed group -> stop
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # Step 4: recompute each centroid as the mean of its members.
            # An empty cluster keeps its old centroid (rare, but crashes
            # naive implementations with a mean-of-nothing warning).
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    centers[k] = members.mean(axis=0)

        # Inertia = within-cluster sum of squared distances (what the
        # elbow method plots)
        inertia = ((X - centers[labels]) ** 2).sum()
        return centers, labels, inertia, n_iter

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
