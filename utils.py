import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer

import numpy as np


def mapValue(oldMin, oldMax, newMin, newMax, value):
    """
    Mapping of number from a range to another
    """
    OldRange = oldMax - oldMin
    NewRange = newMax - newMin
    new_value = (((value - oldMin) * NewRange) / OldRange) + newMin
    return new_value


def createNet(n_inputs, n_outputs, n_hidden_layers=0, n_hidden_layer_units=100, use_dropout=False, dropout_perc=0.3,
              nonlinear=torch.nn.Tanh):
    """
    Function to create simple fully-connected NNs
    """
    if n_hidden_layers == 0:
        return torch.nn.Linear(n_inputs, n_outputs)
    layers = [torch.nn.Linear(n_inputs, n_hidden_layer_units)]
    for i in range(n_hidden_layers - 1):
        layers.append(nonlinear())
        layers.append(torch.nn.Linear(n_hidden_layer_units, n_hidden_layer_units))
        if use_dropout:
            layers.append(torch.nn.Dropout(p=dropout_perc))

    layers.append(nonlinear())
    layers.append(torch.nn.Linear(n_hidden_layer_units, n_outputs))
    return torch.nn.Sequential(*layers)


def ScaleData(train_x, test_x, scaling_method, dimension, random_seed):
    """
    Scale training and test data according to specified method and dimension, see https://github.com/lpphd/mtsscaling
    """
    if scaling_method == "none":
        return train_x, test_x
    if np.isinf(train_x).any() or np.isinf(test_x).any():
        print("Train or test set contains infinity.")
        exit(-1)
    scaling_dict = {"minmax": MinMaxScaler(), "maxabs": MaxAbsScaler(), "standard": StandardScaler(),
                    "robust": RobustScaler(), "quantile": QuantileTransformer(random_state=random_seed),
                    "powert": PowerTransformer(), 'normalize': Normalizer()}
    if scaling_method not in scaling_dict.keys():
        print(f"Scaling method {scaling_method} not found.")
        exit(-1)
    if dimension not in ['timesteps', 'channels', 'all', 'both']:
        print(f"Dimension {dimension} not found.")
        exit(-1)

    dim1 = -1
    dim2 = 1
    if scaling_method == 'normalize':
        dim1 = 1
        dim2 = -1
    out_train_x = np.zeros_like(train_x, dtype=np.float32)
    out_test_x = np.zeros_like(test_x, dtype=np.float32)
    train_shape = train_x.shape
    test_shape = test_x.shape
    if dimension == 'all':
        out_train_x = scaling_dict[scaling_method].fit_transform(train_x.reshape((dim1, dim2))).reshape(train_shape)
        if scaling_method == 'normalize':
            out_test_x = scaling_dict[scaling_method].fit_transform(test_x.reshape((dim1, dim2))).reshape(test_shape)
        else:
            out_test_x = scaling_dict[scaling_method].transform(test_x.reshape((dim1, dim2))).reshape(test_shape)
    else:
        if dimension == 'channels':
            train_channel_shape = train_x[:, 0, :].shape
            test_channel_shape = test_x[:, 0, :].shape
            for i in range(train_x.shape[1]):
                out_train_x[:, i, :] = scaling_dict[scaling_method].fit_transform(
                    train_x[:, i, :].reshape((dim1, dim2))).reshape(
                    train_channel_shape)
                if scaling_method == 'normalize':
                    out_test_x[:, i, :] = scaling_dict[scaling_method].fit_transform(
                        test_x[:, i, :].reshape((dim1, dim2))).reshape(
                        test_channel_shape)
                else:
                    out_test_x[:, i, :] = scaling_dict[scaling_method].transform(
                        test_x[:, i, :].reshape((dim1, dim2))).reshape(
                        test_channel_shape)
        elif dimension == 'timesteps':
            train_timest_shape = train_x[:, :, 0].shape
            test_timest_shape = test_x[:, :, 0].shape
            for i in range(train_x.shape[2]):
                out_train_x[:, :, i] = scaling_dict[scaling_method].fit_transform(
                    train_x[:, :, i].reshape((dim1, dim2))).reshape(
                    train_timest_shape)
                if scaling_method == 'normalize':
                    out_test_x[:, :, i] = scaling_dict[scaling_method].fit_transform(
                        test_x[:, :, i].reshape((dim1, dim2))).reshape(
                        test_timest_shape)
                else:
                    out_test_x[:, :, i] = scaling_dict[scaling_method].transform(
                        test_x[:, :, i].reshape((dim1, dim2))).reshape(
                        test_timest_shape)
        elif dimension == 'both':
            train_both_shape = train_x[:, 0, 0].shape
            test_both_shape = test_x[:, 0, 0].shape
            for i in range(train_x.shape[1]):
                for j in range(train_x.shape[2]):
                    out_train_x[:, i, j] = scaling_dict[scaling_method].fit_transform(
                        train_x[:, i, j].reshape((dim1, dim2))).reshape(
                        train_both_shape)
                    if scaling_method == 'normalize':
                        out_test_x[:, i, j] = scaling_dict[scaling_method].fit_transform(
                            test_x[:, i, j].reshape((dim1, dim2))).reshape(
                            test_both_shape)
                    else:
                        out_test_x[:, i, j] = scaling_dict[scaling_method].transform(
                            test_x[:, i, j].reshape((dim1, dim2))).reshape(
                            test_both_shape)
        else:
            print(f"Dimension {dimension} not found.")
            exit(-1)
    return out_train_x, out_test_x
