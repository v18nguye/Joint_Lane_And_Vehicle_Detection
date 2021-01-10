import yaml


class CarConfig:
    """Car model configuration

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

    def __repr__(self):
        return self.config_str

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
