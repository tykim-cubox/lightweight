import os, sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np

import torch, detectron2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode

import cv2
import IPython



if __name__ == '__main__':

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    print(cfg.MODEL.DEVICE)
    cfg.MODEL.DEVICE='cuda'
    # sys.exit()

    cfg.MODEL.WEIGHTS = 'output/model_final.pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    from detectron2.data.datasets import register_coco_instances

    # register_coco_instances("my_dataset_val", {}, "runs/labelme2coco/val.json", "hold_smartphone")
    # my_dataset_val_metadata = MetadataCatalog.get("my_dataset_val")

    register_coco_instances("my_dataset_train", {}, "datasets/hold_smartphone-1.json", "datasets/hold_smartphone")
    my_dataset_val_metadata = MetadataCatalog.get("my_dataset_train")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9

    predictor = DefaultPredictor(cfg)

    WIDTH = [1920, 1280][1]
    HEIGHT = [1080, 720][1]
    DO_RECORD = True

    # cap = cv2.VideoCapture('./test.mp4') # Video
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW) # WebCAM
    

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    out_frames = []

    count = 1
    accumulated_time = 0
    while cv2.waitKey(33) < 0:
        ret, frame = cap.read()
        
        if not ret:
            break

        start = time.time()

        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        outputs = predictor(frame)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(frame[:, :, ::-1],
                    metadata=my_dataset_val_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        frame = out.get_image()[..., ::-1]

        accumulated_time += (time.time() - start)
        if count % 10 == 0:
            print(f'FPS: {1 / (accumulated_time / 10):.2f}')
            accumulated_time = 0
        count += 1

        if DO_RECORD:
            out_frames.append(frame[..., ::-1])
        cv2.imshow("frame", frame)

    cap.release()
    cv2.destroyAllWindows()

    if DO_RECORD:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        clips = ImageSequenceClip(out_frames, fps=30)
        clips.write_videofile('out.mp4', threads=4)