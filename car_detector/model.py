import torch

from lib.models import Darknet
from lib.utils.utils import non_max_suppression, load_classes


class CarDetector:
    """ Car detection model

    :param cfg: yaml's object
        model config object

    :param device:
            GPU devices

    :reference:
        Car detection model based on the YOLOv3 architecture:
            https://github.com/eriklindernoren/PyTorch-YOLOv3

    """

    def __init__(self, cfg, device):
        self.model = None
        self.coco = load_classes(cfg['classes'])  # classes in the CoCo data-set
        self.cfg = cfg
        self.device = device
        self.load_model()

    def load_model(self):
        """Load a pre-trained model"""
        self.model = Darknet(self.cfg['model']['type'], self.cfg['im_size']).to(self.device)
        self.model.load_darknet_weights(self.cfg['model']['weights'])
        self.model.eval()

    def detect(self, im):
        """Car detection on an image or a frame"""
        im = im.to(self.device)
        with torch.no_grad():
            detection = self.model(im)
            detection = non_max_suppression(detection, self.cfg['conf_thres'],
                                            self.cfg['nms_thres'])

        return detection
