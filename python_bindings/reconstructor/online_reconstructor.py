#!/usr/bin/env python
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import os
import rospy
from cv_bridge import CvBridge
import torch

class online_reconstructor:
    def __init__(self) -> None:
        
        if torch.cuda.is_available():
            print("CUDA is available")
            current_device = torch.cuda.current_device()
            print(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
        else:
            print("CUDA is not available")

        package_path = os.path.dirname(os.path.abspath(__file__))
        self.__model = load_model(package_path + '/pretrained/firenet_1000.pth.tar')
        self.__device = get_device(use_gpu=True)
        self.__model = self.__model.to(self.__device)
        self.__model.eval()

        parser = argparse.ArgumentParser(description='Evaluating a trained network')
        set_inference_options(parser)
        self.args = parser.parse_args()

        self.args.use_gpu = True

        self.index = 0
        
        self.reconstructor = ImageReconstructor(self.__model, 480, 640, 5, self.args)

    def process_events(self, events):
        event_window = np.zeros((4, len(events)))
        for i, event in enumerate(events):
            event_window[0, i] = event["t"]
            event_window[1, i] = event["x"]
            event_window[2, i] = event["y"]
            event_window[3, i] = event["p"]

        last_timestamp = event_window[0, -1]

        voxel_grid = events_to_voxel_grid_pytorch(event_window.T, num_bins=5, width=640, height=480, device=self.__device)
        out = self.reconstructor.update_reconstruction(voxel_grid, self.index, last_timestamp)
        self.index += 1
        return out


