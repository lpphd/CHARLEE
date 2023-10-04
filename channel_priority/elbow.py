# ElbowCut
import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

from channel_priority.shrunk_cent import shrunk_centroid
from channel_priority.calc_distance import distance_matrix
from channel_priority.utils import detect_knee_point
from sklearn.base import TransformerMixin, BaseEstimator


class elbow(TransformerMixin, BaseEstimator):

    def fit(self, X, y):
        centroid_obj = shrunk_centroid(0)
        df = centroid_obj.create_centroid(X.copy(), y)
        obj = distance_matrix()
        self.distance_frame = obj.distance(df)

        self.relevant_dims = []
        distance = self.distance_frame.sum(axis=1).sort_values(ascending=False).values
        indices = self.distance_frame.sum(axis=1).sort_values(ascending=False).index
        self.ranked_dims = indices.values

        self.relevant_dims.extend(detect_knee_point(distance, indices)[0])

        return self

    def transform(self, X):
        return X.iloc[:, self.relevant_dims]
