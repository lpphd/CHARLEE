import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
import pandas as pd
from channel_priority.shrunk_cent import shrunk_centroid
from channel_priority.calc_distance import distance_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class kmeans(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)
        # l2 normalisng for kmeans
        self.distance_frame = pd.DataFrame(normalize(self.distance_frame, axis=0),
                                           columns=self.distance_frame.columns.tolist())

        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(self.distance_frame)
        # Find the cluster name with maximum avg distance
        self.cluster = np.argmax(self.kmeans.cluster_centers_.mean(axis=1))
        self.ranked_dims = np.argsort(
            np.linalg.norm(self.kmeans.cluster_centers_[self.cluster] - self.distance_frame.values, axis=1))
        self.relevant_dims = [id_ for id_, item in enumerate(self.kmeans.labels_) if item == self.cluster]

        return self

    def transform(self, X):
        return X.iloc[:, self.relevant_dims]
