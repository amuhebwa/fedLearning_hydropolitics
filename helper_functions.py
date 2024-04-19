mport
code

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from hydroeval import *
from hydroeval import evaluator, nse, kge
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from utils import *

# check if path exists
def check_path_exists(path):
    if not os.path.exists(path):
        raise Exception(f"Path {path} does not exist")
    return True


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def calculate_mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    return np.mean(np.square(_error(actual, predicted)))


# Ref: https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(calculate_mse(actual, predicted))


def calculate_KGE(actual, predicted):
    # Convert inputs to numpy arrays and flatten them
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    # Handle infinities by replacing them with NaNs
    actual = np.where(np.isinf(actual), np.nan, actual)
    predicted = np.where(np.isinf(predicted), np.nan, predicted)

    # Filter out NaN and inf values
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    # Compute KGE
    my_kge = evaluator(kge, predicted, actual)
    return my_kge[0][0]


# BOUNDED ORIGINAL NSE
def calculate_NSE(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    my_nse = evaluator(nse, predicted, actual)
    return my_nse[0]


def calculate_RBIAS(actual, predicted):
    actual = np.asarray(actual.ravel())
    predicted = np.asarray(predicted.ravel())
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    rbias = np.nanmean((predicted - actual) / np.nanmean(actual))
    return rbias


"""
def calculate_NRMSE(actual: np.ndarray, predicted: np.ndarray):
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]
    return rmse(actual, predicted) / (actual.max() - actual.min())
"""


def calculate_NRMSE(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    # Get indices where neither actual nor predicted are NaN
    valid_indices = ~np.isnan(actual) & ~np.isnan(predicted)
    # Filter out invalid indices (where either value is NaN)
    actual = actual[valid_indices]
    predicted = predicted[valid_indices]

    # Check if the filtered array is empty
    if actual.size == 0:
        raise ValueError("No valid data points after removing NaN values.")

    return rmse(actual, predicted) / (actual.max() - actual.min())


def create_dataset_forecast(_dataset, n_steps_in: int, n_steps_out: int):
    """
    format the dataset to be used in the LSTM model
    """
    X, y = list(), list()
    for i in range(len(_dataset)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        if out_end_ix > len(_dataset):
            break
        seq_x, seq_y = _dataset[i:end_ix, :-1], _dataset[end_ix - 1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def load_and_process_dataset(src_path, num_of_years):
    """
    load and process each of the region's dataset
    """
    dataset_df = pd.read_csv(src_path)
    # drop date column and original_streamflow
    dataset_df.drop(columns=['Date', 'original_streamflow'], inplace=True)
    # stat_features = [col for col in dataset_df.columns if dataset_df[col].nunique() == 1]
    # dynamic_features = [col for col in dataset_df.columns if (dataset_df[col].nunique() > 1) & (col != 'date')]

    # format streamflow
    dataset_df = dataset_df[~dataset_df['streamflow'].isna()]
    # process static data
    static_df = dataset_df[stat_features]
    static_df += (np.random.rand(*static_df.shape)) * 0.01  # add a small amount of noise to the data
    static_df = minmax_scale(static_df.to_numpy(), feature_range=(0, 1), axis=1, copy=True)
    static_df = pd.DataFrame(static_df, columns=stat_features)
    dataset_df[stat_features] = static_df[stat_features]

    '''
    For optimization, we might need to use only 10 years of data.
    Once we have a stronger machine, we will use all the data.
    '''
    dataset_df = dataset_df.iloc[-(365 * num_of_years):, :]
    # process dynamic data
    scalers = dict()
    for _, current_column in enumerate(dynamic_features):
        current_scaler = MinMaxScaler(feature_range=(0, 1))
        scalers['scaler_' + str(current_column)] = current_scaler
        dataset_df[current_column] = (current_scaler.fit_transform(dataset_df[current_column].values.reshape(-1, 1))).ravel()
    return dataset_df, scalers


def average_lstm_weights(lstm_models):
    # Initialize a variable to store the sum of weights
    sum_of_weights = [tf.Variable(tf.zeros_like(w)) for w in lstm_models[0].get_weights()]

    # Iterate over models and accumulate weights
    for model in lstm_models:
        model_weights = model.get_weights()
        sum_of_weights = [tf.add(sum_w, w) for sum_w, w in zip(sum_of_weights, model_weights)]

    # Calculate the average weights
    num_models = len(lstm_models)
    averaged_weights = [tf.divide(sum_w, num_models) for sum_w in sum_of_weights]
    return averaged_weights
