import _init_paths
import os
import argparse

import tqdm
import cv2
import torch

from annotator import lane_ann, car_ann
from pre_processor import lane_prx2, car_prx2

from car_detector.config import CarConfig
from car_detector.model import CarDetector
from lane_detector.config import LaneConfig
from lane_detector.model import LaneDetector


def parse_args():
    """Argument Parser"""
    parser = argparse.ArgumentParser(description="Car Lane Joint Detection")
    parser.add_argument("-m", "--mode", choices=["image", "video"], default="image")
    parser.add_argument("--fps", type=int, default=20, help='registered frames-per-second for videos')
    parser.add_argument("-dp", "--data_path", default="data/images",
                        help="path to an image directory or a explicit path to a video")
    parser.add_argument("-lcf", "--lane_cfg_path", default="cfgs/lane.yml",
                        help="Path to lane-model-config file")
    parser.add_argument("-ccf", "--car_cfg_path", default="cfgs/car.yml",
                        help="Path to car-model-config file")
    parser.add_argument("-odr", "--out_dir", default="output",  help="Saving directory")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create the output dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Config torch device
    device = torch.device('cuda')

    # Load a pre-trained lane detection model
    print('Loading lane detection model and its configuration: ', args.lane_cfg_path)
    lane_cfg = LaneConfig(args.lane_cfg_path)
    lane = LaneDetector(lane_cfg, device)

    # Load a pre-trained car detection model
    print('Loading car detection model and its configuration: ', args.lane_cfg_path)
    car_cfg = CarConfig(args.car_cfg_path)
    car = CarDetector(car_cfg, device)

    if args.mode == "image":
        # Load a list of images
        print('Loading images: ', args.data_path)
        images = os.listdir(args.data_path)

        # Run car, lane detections
        print('Running detection on images ...')
        for idx, item in enumerate(tqdm.tqdm(images)):
            # Read image
            im = cv2.imread(os.path.join(args.data_path, item))

            # Process the image for lane detection
            lane_im = lane_prx2(im, lane_cfg['model']['parameters']['img_h'],
                                lane_cfg['model']['parameters']['img_w'])

            # Process the image for car detection
            car_im = car_prx2(im, car_cfg['im_size'])

            # Running detection on the processed image
            lane_pred = lane.detect(lane_im)[0]
            car_pred = car.detect(car_im)[0]

            # Annotate the prediction
            ann_im, lines = lane_ann(im, lane_pred)
            ann_im = car_ann(ann_im, car_pred, car_cfg, car.coco, lines)

            # Save prediction
            cv2.imwrite(args.out_dir + '/sav_' + images[idx], ann_im)

    if args.mode == "video":
        # Load a video and get its size
        print('Loading video: ', args.data_path)
        cap = cv2.VideoCapture(args.data_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        # Create output-video writer
        print('Creating a video writer')
        out = cv2.VideoWriter('output/sav_' + args.data_path.split('/')[-1].split('.')[0] + '.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              args.fps, size)

        print('Running detection on video ...')
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:

                # Process frame for lane detection
                lane_frame = lane_prx2(frame, lane_cfg['model']['parameters']['img_h'],
                                       lane_cfg['model']['parameters']['img_w'])

                # Process frame for car detection
                car_frame = car_prx2(frame, car_cfg['im_size'])

                # Running detection on the processed image
                lane_pred = lane.detect(lane_frame)[0]
                car_pred = car.detect(car_frame)[0]

                # Annotate the prediction
                ann_frame, lines = lane_ann(frame, lane_pred)
                ann_frame = car_ann(ann_frame, car_pred, car_cfg, car.coco, lines)

                # write the annotated frame
                out.write(ann_frame)

                # cv2.imshow('frame', lane_frame)       # Un-comment three lines to observe processed videos
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print('Saving results to: ', args.out_dir)
    print('Finish !')


if __name__ == '__main__':
    main()
