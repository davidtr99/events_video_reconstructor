#!/usr/bin/env python
import dvs_msgs.msg

from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import os

class online_reconstructor:
    def __init__(self) -> None:
        package_path = os.path.dirname(os.path.abspath(__file__))
        model = load_model(package_path + '/pretrained/firenet_1000.pth.tar')
        device = get_device(use_gpu=True)
        model = model.to(device)
        model.eval()
        print('Model loaded.')

    def process_events(self, events):
        print('> Size of events: ', len(events))

    def test(self):
        print('test')


