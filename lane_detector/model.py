import os
import random
import numpy as np
import torch


class LaneDetector:
    """ Lane detection model

    :param cfg: yaml's object
        model config object

    :reference:
        Lane detection model based on the LaneATT architecture:
            https://github.com/lucastabelini/LaneATT

    """
    def __init__(self, cfg, device):
        # load a config file
        self.cfg = cfg
        self.model = None
        self.device = device
        self.load_model()
        # enable the benchmarking feature
        torch.backends.cudnn.benchmark = True
        # fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

    def load_model(self):
        """Load a pre-trained model"""
        model_path = os.path.join(self.cfg['model']['path'])
        model = self.cfg.get_model()
        # load model's parameters from a dict
        model.load_state_dict(torch.load(model_path)['model'])
        # enable GPU
        self.model = model.to(self.device)
        # set model to the eval mode
        self.model.eval()

    def detect(self, im):
        """Running lane detection on an image or a frame"""
        parameters = self.cfg.get_test_parameters()
        with torch.no_grad():
            im = im.to(self.device)
            output = self.model(im, **parameters)
            prediction = self.model.decode(output, as_lanes=True)

        return prediction
