import numpy as np
import pandas as pd
import Geohash

import helpers.constants as constants


def load_raw_data(data_path):
    upper_df = pd.read_csv('{}/upper_half_data.csv'.format(data_path), index_col=0)
    lower_df = pd.read_csv('{}/lower_half_data.csv'.format(data_path), index_col=0)

    # Return the concatenated dataframe
    return pd.concat([upper_df, lower_df])


def decode_geohash(geo_str, use_minmax_norm=True):
    geo_tup = Geohash.decode(geo_str)
    if use_minmax_norm:
        return ((geo_tup[0] - constants.geohash_min_X) / (constants.geohash_max_X - constants.geohash_min_X),
                (geo_tup[1] - constants.geohash_min_Y) / (constants.geohash_max_Y - constants.geohash_min_Y))
    else:
        return geo_tup


def convert_day_to_one_hot_feature(day):
    # Day starts from 1
    return convert_to_one_hot((day - 1) % 7, num_categories=7)


def convert_total_minutes_to_cyclical_feature(total_minutes):
    sin_feature = np.sin(2 * np.pi * total_minutes / 1440)
    cosine_feature = np.cos(2 * np.pi * total_minutes / 1440)
    return np.array([sin_feature, cosine_feature])


def convert_total_minutes_to_part_of_day_one_hot_feature(total_minutes):
    # Morning: 7:30 (450) to 10:30 (630)
    # Midday: 10:45 (645) to 13:00 (780)
    # Afternoon: 13:15 (795) to 17:30 (1050)
    # Evening: 17:45 (1065) to 20:00 (1200)
    # Night: 20:15 (1215) to 00:00 (0)
    # Midnight: 00:15 (15) to 7:15 (435)
    idx = None
    if 450 <= total_minutes <= 630:
        idx = 0    # Morning
    if 645 <= total_minutes <= 780:
        idx = 1    # Midday
    if 795 <= total_minutes <= 1050:
        idx = 2    # Afternoon
    if 1065 <= total_minutes <= 1200:
        idx = 3    # Evening
    if total_minutes >= 1215 or total_minutes == 0:
        idx = 4    # Night
    if 15 <= total_minutes <= 435:
        idx = 5    # Midnight
    return convert_to_one_hot(idx, 6)


def convert_to_one_hot(int_value, num_categories):
    # The integer starts from zero
    one_hot_np = np.zeros(num_categories)
    np.put(one_hot_np, int_value, 1)
    return one_hot_np


def compute_total_minutes(timestring):
    hour, minute = [int(value) for value in timestring.split(':')]
    total_minutes = hour * 60 + minute
    return hour * 60 + minute


def get_sliding_features(seq, sliding_len=3, sliding_stride=2):
    stride = seq.strides[0]
    shape = [(len(seq) - sliding_len)//sliding_stride + 1, sliding_len]
    sequence_strides = np.lib.stride_tricks.as_strided(seq, shape=shape, strides=[sliding_stride*stride, stride])
    return sequence_strides


def negative_softmax(adj_np):
    exp = np.exp(-adj_np)
    return exp / exp.sum(axis=1)[:, np.newaxis]


def normalize_adj_matrix(adj_np):
    """
    Normalize adjacency matrix (with renormalization)
    """
    d = np.sum(adj_np, axis=1)
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    return adj_np.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj_matrix(adj_np, use_znorm, epsilon):
    num_geohash = adj_np.shape[0]

    # Z-normalization
    if use_znorm:
        adj_np = (adj_np - adj_np.mean()) / adj_np.std()

    # Filter out pairs of geohash with large distance for sparsity
    adj_np[adj_np >= epsilon] = np.inf

    adj_np = negative_softmax(adj_np)

    # Set diagonal to one
    adj_np[np.arange(num_geohash), np.arange(num_geohash)] = 1
    
    return normalize_adj_matrix(adj_np)
