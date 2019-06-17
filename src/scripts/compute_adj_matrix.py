"""
Compute adjacency matrix of geohash pairs uding l2 distance, and save as numpy file
"""

import numpy as np
import pandas as pd

from helpers.data_utils import decode_geohash


if __name__ == '__main__':
    geohash_df = pd.read_csv('data_loader/data/geohash.csv', index_col=0)

    geohash_ls = geohash_df['geohash6'].tolist()
    decoded_geohash_ls = [decode_geohash(geohash, use_minmax_norm=False) for geohash in geohash_ls]    # Decode
    num_geohash = len(geohash_ls)
    
    # Create adjacency matrix
    adj_np = np.zeros((len(geohash_ls), len(geohash_ls)))

    for i in range(num_geohash):
        for j in range(i, num_geohash):
            if i == j:
                # Set diagonal to zero
                continue
            else:
                l2_dist = np.linalg.norm(np.array(decoded_geohash_ls[i]) - np.array(decoded_geohash_ls[j]))
                adj_np[i, j] = l2_dist
                adj_np[j, i] = l2_dist

    np.save('data_loader/data/l2_dist_adj_mat.npy', adj_np)
