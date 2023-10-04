import pandas as pd
import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.append("..")

from channel_priority.utils import *

from sktime.datatypes._panel._convert import from_nested_to_3d_numpy, from_3d_numpy_to_nested
from sklearn.neighbors import NearestCentroid


class shrunk_centroid:

    def __init__(self, shrink):
        self.shrink = shrink

    def create_centroid(self, X, y):
        """
        Creating the centroid for each class
        """

        # y = X.class_vals
        # X.drop('class_vals', axis = 1, inplace = True)
        cols = X.columns.to_list()
        ts = from_nested_to_3d_numpy(X)  # Contains TS in numpy format
        centroids = []

        for dim in range(ts.shape[1]):
            train = ts[:, dim, :]
            clf = NearestCentroid(shrink_threshold=self.shrink)
            clf.fit(train, y)
            centroids.append(clf.centroids_)

        centroid_frame = from_3d_numpy_to_nested(np.stack(centroids, axis=1), column_names=cols)
        centroid_frame['class_vals'] = clf.classes_

        return centroid_frame.reset_index(drop=True)
