#!/usr/bin/env python3
from utils.loading_utils import load_model, get_device
import numpy as np
from utils.inference_utils import events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
import os
import torch
import yaml

class online_reconstructor:
    def __init__(self) -> None:

        # Check if CUDA is available
        if torch.cuda.is_available():
            print("CUDA is available! ")
            current_device = torch.cuda.current_device()
            print(f"Using GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
        else:
            print("CUDA is not available")

        # Paths
        python_package_path = os.path.dirname(os.path.abspath(__file__))
        ros_package_path = os.path.dirname(os.path.dirname(python_package_path))

        # Load paramaters
        self.__params = load_params_from_file(ros_package_path + '/config/inference_params.yaml')

        # Load the model
        self.__model = load_model(python_package_path + '/pretrained/' + self.__params['trained_model'])
        self.__device = get_device(use_gpu=True)
        self.__model = self.__model.to(self.__device)
        self.__model.eval()
        self.__reconstructor = ImageReconstructor(self.__model, self.__params['camera_height'], self.__params['camera_width'], 5, self.__params)

        # Initialize the general index of the reconstruction
        self.__index = 0

    def process_events(self, events_x, events_y, events_ts, events_p):
        event_window = np.zeros((4, len(events_x)))
        event_window[0, :] = events_ts
        event_window[1, :] = events_x
        event_window[2, :] = events_y
        event_window[3, :] = events_p
        last_timestamp = event_window[0, -1]
        
        voxel_grid = events_to_voxel_grid_pytorch(event_window.T, num_bins=5, width=self.__params['camera_width'], height=self.__params['camera_height'], device=self.__device)
        out = self.__reconstructor.update_reconstruction(voxel_grid, self.__index, last_timestamp)
        self.__index += 1
        return out
    

    def catch_events(self, events):
        print(f"Received {len(events)} events")
        print(events[0])

def load_params_from_file(file_path):
    with open(file_path, 'r') as f:
        params = yaml.safe_load(f)
    return params



