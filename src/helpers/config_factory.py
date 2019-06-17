import importlib


class ConfigFactory(object):
    """
    Helper class to instantiate class object with desired constructor arguments
    """
    class_map_dict = {
        'multi_lightgbm_trainer': ('trainers', 'MultiLightGBMTrainer'),
        'nn_trainer': ('trainers', 'NNTrainer'),
        'multi_lightgbm': ('models', 'MultiLightGBM'),
        'mlp': ('models', 'MLP'),
        'lstm': ('models', 'LSTM'),
        'tcn': ('models', 'TCN'),
        'spatio_tcn': ('models', 'SpatioTCN'),
    }

    @staticmethod
    def create_instance(name, *args):
        module_name, class_name = ConfigFactory.class_map_dict[name]
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(*args)
