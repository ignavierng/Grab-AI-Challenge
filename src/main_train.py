import sys
import argparse
import logging
from pytz import timezone
from datetime import datetime
import numpy as np

from helpers.config_utils import load_yaml_config, save_yaml_config
from helpers.config_factory import ConfigFactory
from helpers.log_helper import LogHelper
from helpers.dir_utils import create_dir
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
    return parser.parse_args(args=sys.argv[1:])


if __name__ == '__main__':
    args = get_args()
    
    # Load yaml configuration file
    config = load_yaml_config(args.config_path)

    # Setup for logging
    if config['general']['log_enabled']:
        output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
        create_dir(output_dir)
        LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                        level_str=config['general']['log_level'])

        # Save the configuration for logging purpose
        save_yaml_config(config, path='{}/config.yaml'.format(output_dir))

    _logger = logging.getLogger(__name__)

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

    # Get dataset object
    dataset = Dataset(**config['dataset'])
    _logger.info('Finished generating dataset.')
    _logger.info('Dataset info: {}'.format(dataset.info))

    # Create model and log its summary
    model = ConfigFactory.create_instance(config['model']['name'],
                                          config['model']['params'],
                                          adj_np)
    model.print_summary(print_func=model.logger.info)

    # Create trainer and train
    trainer = ConfigFactory.create_instance(config['trainer']['name'],
                                            model,
                                            dataset,
                                            config['trainer']['params'],
                                            output_dir)
    trainer.train()
