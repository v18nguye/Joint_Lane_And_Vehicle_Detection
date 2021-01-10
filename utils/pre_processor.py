"""
Pre-processing Functions

"""
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ToTensor


def resize(image, size):
    """Resize images"""
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def lane_prx2(im, im_h=360, im_w=640):
    """ pre-processing images for lane detector

    :param im: ndarray
            an image or a frame on which lanes are shown

    :param im_h: int
            re-scaled image's height for the detector

    :param im_w: int
            re-scaled image's width for the detector

    :return:
        proc_im: tensor with a shape (1, n_channel, im_h, im_w)
            processed image

    """

    # to_tensor = ToTensor()
    im_rz = cv2.resize(im, (im_w, im_h)) / 255
    proc_im = ToTensor()(im_rz.astype(np.float32))
    proc_im = torch.unsqueeze(proc_im, 0)

    return proc_im


def car_prx2(im, im_size):
    """ pre-processing images for car detector

    :param im: ndarray
            an image or a frame on which lanes are shown

    :param im_size: int
            re-scaled image's size for the detector

    :return:
        proc_im: tensor with a shape (1, n_channel, im_size, im_size)
            processed image

    """
    # Convert to Pytorch tensor
    proc_im = ToTensor()(im)
    # Pad to square resolution
    proc_im, _ = pad_to_square(proc_im, 0)
    # Resize
    proc_im = resize(proc_im, im_size)

    proc_im = torch.unsqueeze(proc_im, 0)
    return proc_im
