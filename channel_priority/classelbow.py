# Elbow with class pairwise
import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

import numpy as np
from channel_priority.shrunk_cent import shrunk_centroid
from channel_priority.calc_distance import distance_matrix
from channel_priority.utils import detect_knee_point
from sklearn.base import TransformerMixin, BaseEstimator


class ElbowPair(TransformerMixin, BaseEstimator):
    """
    Class of extract dimension from each class pair
    inp: Shrinkage

    """

    def __init__(self, shrinkage=0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        centroid_obj = shrunk_centroid(self.shrinkage)
        df = centroid_obj.create_centroid(X.copy(), y)
        # print("Centroid Shape: ", df.shape)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        self.ranked_dims_list = []
        for pairdistance in self.distance_frame.iteritems():
            distance = pairdistance[1].sort_values(ascending=False).values
            indices = pairdistance[1].sort_values(ascending=False).index
            self.ranked_dims_list.append(indices.values)

            self.relevant_dims.extend(detect_knee_point(distance, indices)[0])
            self.relevant_dims = list(set(self.relevant_dims))

        self.ranked_dims_list = np.array(self.ranked_dims_list)
        average_ranks = []
        for d in range(self.ranked_dims_list.shape[1]):
            average_ranks.append((self.ranked_dims_list.shape[1] - np.argmax(self.ranked_dims_list == d)).mean())

        self.ranked_dims = np.argsort(average_ranks)[::-1]

        return self

    def transform(self, X):
        return X.iloc[:, self.relevant_dims]
