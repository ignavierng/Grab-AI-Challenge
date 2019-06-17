import yaml


def load_yaml_config(path):
    with open(path, 'r') as stream:
        return yaml.safe_load(stream)


def save_yaml_config(config, path):
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
