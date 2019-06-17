import sys
import argparse
import numpy as np
import pandas as pd

from helpers.config_utils import load_yaml_config
from helpers.config_factory import ConfigFactory
from helpers.data_utils import preprocess_adj_matrix
from helpers.tf_utils import set_seed
from data_loader.dataset import Dataset


def get_args():
    # TODO: Add parser for CUDA_VISIBLE_DEVICES
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        type=str,
                        required=True,
                        help='Path of YAML configuration file')

    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='Path of directory containing model weights')

    parser.add_argument('--inference_data_path',
                        type=str,
                        required=True,
                        help='Path of test data for inference (csv file)')

    return parser.parse_args(args=sys.argv[1:])


if __name__ == '__main__':
    args = get_args()

    # Load yaml configuration file
    config = load_yaml_config(args.config_path)

    # Reproducibility
    set_seed(config['general']['seed'])

    # Construct adjacency matrix
    if 'adj_mat' in config:
        adj_np = np.load(config['adj_mat']['path'])
        adj_np = preprocess_adj_matrix(adj_np,
                                       config['adj_mat']['use_znorm'],
                                       config['adj_mat']['epsilon'])
    else:
        adj_np = None

    # Load test data for inference
    data_df = pd.read_csv(args.inference_data_path, index_col=0)
    geohash_df = pd.read_csv('data_loader/data/geohash.csv', index_col=0)
    X_np, y_np \
        = Dataset.generate_features_and_labels_for_inference(data_df,
                                                             geohash_df,
                                                             num_steps=config['dataset']['num_steps'],
                                                             batch_size=config['dataset']['batch_size'],
                                                             sliding_features=config['dataset']['sliding_features'],
                                                             use_geohash=config['dataset']['use_geohash'],
                                                             use_day=config['dataset']['use_day'],
                                                             use_cyclical_timestamp=config['dataset']['use_cyclical_timestamp'],
                                                             use_part_of_day=config['dataset']['use_part_of_day'])

    # Create model and load its weights
    model = ConfigFactory.create_instance(config['model']['name'],
                                          config['model']['params'],
                                          adj_np)
    model.load(args.model_path)
    model.print_summary(print_func=print)

    # Inference
    print('Begin inferencing.')
    pred_np, rmse = model.predict(X_np, y_np)
    print('rmse: {}'.format(rmse))
    print('pred_np.shape: {}'.format(pred_np.shape))
