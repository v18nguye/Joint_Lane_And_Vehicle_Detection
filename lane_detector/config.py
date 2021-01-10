import yaml
import lane_detector.lib.models as models


class LaneConfig:
    """Lane model configuration

    :param path: string
        config file path

    """

    def __init__(self, path):
        self.config = {}
        self.config_str = ""
        self.load(path)

    def load(self, path):
        with open(path, 'r') as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def get_model(self, **kwargs):
        name = self.config['model']['name']
        parameters = self.config['model']['parameters']
        return getattr(models, name)(**parameters, **kwargs)

    def get_test_parameters(self):
        return self.config['test_parameters']

    def __repr__(self):
        return self.config_str

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config