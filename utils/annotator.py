"""
Annotating Functions
"""
import random

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


def to_numpy(x):
    """Convert tensors to numpy arrays"""
    x = torch.Tensor.cpu(x).detach().numpy()

    return int(x)


def lin_f(xy_s, xy_e):
    """Find a function representing for a line:
                    y = ax + b

    :param: xy_s: tuple
        start point coordinate of the line
    :param: xy_s: tuple
        end point coordinate of the line

    """
    a = (xy_e[1] - xy_s[1]) / (xy_e[0] - xy_s[0])
    b = xy_e[1] - a * xy_e[0]

    return a, b


def Z_point(xy_g, a, b):
    """Find a Z point so that  GZ is
        perpendicular to the line defined by y = ax + b
    :param: xy_g: tuple
        G point's coordinate

    :param: a: float
    :param: b: float

    """
    x_z = (xy_g[0] + a * xy_g[1] - a * b) / (1 + a ** 2)
    y_z = a * x_z + b

    # Distance GZ
    dist = np.sqrt((xy_g[1] - y_z) ** 2 + (xy_g[0] - x_z) ** 2) * np.sign(x_z)

    xy_z = (x_z, y_z)

    return xy_z, dist


def border_label(pred, im_w, im_h):
    """Label border from 0 to N depending on its position
        to the origin (x = 0, y = im_h).
        0:  the first border line
        N: the last border line

    """
    lines = {}

    point_set = []
    dist_set = []
    func_set = []  # line functions
    s_point_set = []    # line's start point
    e_point_set = []    # line's end point

    for i, lane in enumerate(pred):
        points = lane.points
        N = points.shape[0]
        points[:, 0] *= im_w
        points[:, 1] *= im_h
        points = points.round().astype(int)

        # plot line between start and end points
        s_point = tuple(points[0, :])
        e_point = tuple(points[-1, :])
        s_point_set.append(s_point)
        e_point_set.append(e_point)

        # find a function, y = ax +b, represents for the line
        a, b = lin_f(s_point, e_point)

        # find line distance to the origin (0,im_h)
        # _, dist = Z_point((0, im_h-1), a, b)

        point_set.append(points)
        # dist_set.append(dist)
        dist_set.append(e_point[0])     # end-point's x coordinate
        func_set.append((a, b))

    sort_inds = np.argsort(dist_set)
    for i, ind in enumerate(list(sort_inds)):
        lines[i] = {}
        lines[i]['points'] = point_set[ind]
        lines[i]['func'] = func_set[ind]
        lines[i]['code'] = i
        lines[i]['dist'] = dist_set[ind]
        lines[i]['s_point'] = s_point_set[ind]
        lines[i]['e_point'] = e_point_set[ind]

    return lines


def lane_ann(im, pred, color=(0, 255, 0)):
    """ Lane annotation

    :param im: ndarray
            an image or a frame on which lanes are shown

    :param pred: lane's object
            lane's prediction given by models

    :param color: tuple
            lane's color

    :return:
        ann_img: ndarray of the image's shape
            annotated image

    """
    ann_im = im  # initialize

    lines = border_label(pred, im.shape[1], im.shape[0])

    for _, line in lines.items():

        # plot predicted points by the model
        for curr_p, next_p in zip(line['points'][:-1], line['points'][1:]):
            ann_im = cv2.line(im,
                              tuple(curr_p),
                              tuple(next_p),
                              color=color,
                              thickness=3)

        # plot border coding
        xy_G = (int((line['e_point'][0] + line['s_point'][0])/2),
                int((line['e_point'][1] + line['s_point'][1])/2))
        text = 'border: '+str(line['code'])
        text_size = cv2.getTextSize(text,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, 1)

        fontScale = min(int(text_size[0][0] / 2), int(text_size[0][1] / 2)) / (23.5 / 1)
        ann_im = cv2.putText(ann_im, text, xy_G,
                             cv2.FONT_HERSHEY_SIMPLEX, fontScale/1.3, (0, 0, 255), 1)

    return ann_im, lines


def box_plot(cls, conf, x_tl, y_tl, x_br, y_br, im, color_):
    """Bounding box plot

    :param cls: string
        class name

    :param conf: float
        prediction confidence

    :param x_tl: float
        x-top-left corner

    :param y_tl: float
        y-top-left corner

    :param x_br: float
        x-bottom-right corner

    :param y_br: float
        y-bottom-right corner

    :param im: ndarray
        image on which the box is annotated

    :pram color_: tuple
        box's color

    :return:
        ann_img: ndarray
            annotated image
    """

    color = tuple([int(i * 255) for i in color_])

    if torch.is_tensor(x_tl):
        xtl, ytl, xbr, ybr = to_numpy(x_tl), to_numpy(y_tl), to_numpy(x_br), to_numpy(y_br)
    else:
        xtl, ytl, xbr, ybr = x_tl, y_tl, x_br, y_br

    im = cv2.rectangle(im, (xtl, ytl), (xbr, ybr), color, 2)
    text = cls + ' ' + str(int(conf * 100)) + '%'
    text_size = cv2.getTextSize(text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, 1)
    _x1 = xtl
    _y1 = ytl
    _x2 = _x1 + int(text_size[0][0] / 2)
    _y2 = ytl + int(text_size[0][1] / 2)
    cv2.rectangle(im, (_x1, _y1), (_x2, _y2), color, cv2.FILLED)
    fontScale = min(int(text_size[0][0] / 2), int(text_size[0][1] / 2)) / (23.5 / 1)
    cv2.putText(im, text, (xtl, ytl + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1)

    return im


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def verify_lane(xy_c, lanes):
    """Verify which lane a given point C belongs to

    """

    lane_num = None

    dist_set = []  # distances from the point to lane's borders
    lane_codes = []
    func_set = []

    for _, lane in lanes.items():
        z_pt, _ = Z_point(xy_c, lane['func'][0], lane['func'][1])
        dist = np.linalg.norm(np.asarray(xy_c)-np.asarray(lane['s_point']))
        dist_set.append(dist)
        func_set.append(lane['func'])
        lane_codes.append(lane['code'])

    sort_inds = list(np.argsort(dist_set))
    func_1 = func_set[sort_inds[0]]
    sign_1 = np.sign(xy_c[1] - func_1[0] * xy_c[0] - func_1[1])
    func_2 = func_set[sort_inds[1]]
    sign_2 = np.sign(xy_c[1] - func_2[0] * xy_c[0] - func_2[1])

    if sign_1 * sign_2 > 0:
        lane_num = min(lane_codes[sort_inds[0]], lane_codes[sort_inds[1]])

    return lane_num


def lane_label(im, x_tl, y_tl, x_br, y_br, lanes):
    """Label a lane for each car

    """
    ann_im = im
    lane_num = None

    if torch.is_tensor(x_tl):

        xtl, ytl, xbr, ybr = to_numpy(x_tl), to_numpy(y_tl), to_numpy(x_br), to_numpy(y_br)
    else:
        xtl, ytl, xbr, ybr = x_tl, y_tl, x_br, y_br

    # the box's center
    x_c = int((xtl + xbr) / 2)
    y_c = int((ytl + ybr) / 2)
    xy_c = (x_c, y_c)

    ann_im = cv2.drawMarker(ann_im, xy_c, color=(0, 255, 255), markerSize=13, thickness=1)  # plot box's center

    label = 'x'  # if can't find a lane

    if len(lanes.keys()) > 1:
        lane_num = verify_lane(xy_c, lanes)
        if lane_num is not None:
            label = str(lane_num)

    text = 'lane: '+label
    text_size = cv2.getTextSize(text,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, 1)

    fontScale = min(int(text_size[0][0] / 2), int(text_size[0][1] / 2)) / (23.5 / 1)
    ann_im = cv2.putText(ann_im, text, xy_c,
                         cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 255), 1)

    return ann_im, lane_num


def car_ann(im, detection, cfg, coco, lanes=None):
    """Predicted bounding box annotation

    :param im: ndarray
            an image on which bounding boxes are plotted

    :param detection: ndarray
            bounding box prediction given by models

    :param cfg: yaml's object
            model cfg object

    :param coco: dict
            coco classes

    :param lanes: dict
            detected lanes on the image

    :return:
        ann_im: ndarray
            annotated image

    """
    ann_im = im

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # number of cars for each lane
    if lanes is not None:
        num_cars = {k: 0 for k in range(len(lanes.keys()) - 1)}

    # Define relevant classes in the CoCo data-set
    vehicles = ['car', 'truck', 'bus']

    if detection is not None:
        detection = rescale_boxes(detection, cfg['im_size'], im.shape[:2])
        unique_labels = detection[:, -1].cpu().unique()
        n_cls_pred = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_pred)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
            if coco[int(cls_pred)] in vehicles:
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                ann_im = box_plot(coco[int(cls_pred)], cls_conf.item(), x1, y1, x2, y2, im, color)

                # Label a lane for each car
                if lanes is not None:
                    ann_im, lane_num = lane_label(ann_im, x1, y1, x2, y2, lanes)

                    if lane_num is not None:
                        num_cars[lane_num] += 1

    # Labelling number of cars per lane
    if lanes is not None:
        for ind in range(len(lanes.keys())):
            if ind < len(lanes.keys()) - 1:

                x_G = (lanes[ind]['s_point'][0] + lanes[ind]['e_point'][0]
                       + lanes[ind + 1]['s_point'][0] + lanes[ind + 1]['e_point'][0]) / 4
                y_G = (lanes[ind]['s_point'][1] + lanes[ind]['e_point'][1]
                       + lanes[ind + 1]['s_point'][1] + lanes[ind + 1]['e_point'][1]) / 4
                xy_G = (int(x_G), int(y_G))

                text = 'lane '+str(ind)+': '+str(num_cars[ind])
                text_size = cv2.getTextSize(text,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1, 1)
                _x1 = xy_G[0]
                _y1 = xy_G[1]
                _x2 = _x1 + int(text_size[0][0] / 2)
                _y2 = xy_G[1] + int(text_size[0][1] / 2)
                cv2.rectangle(ann_im, (_x1, _y1), (_x2, _y2), (255, 255, 255), cv2.FILLED)
                fontScale = min(int(text_size[0][0] / 2), int(text_size[0][1] / 2)) / (23.5 / 1)
                ann_im = cv2.putText(ann_im, text, (xy_G[0], xy_G[1] + text_size[1]),
                                     cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), 1)

    return ann_im
