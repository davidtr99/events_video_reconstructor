#!/usr/bin/env python3
from utils.loading_utils import load_model, get_device
import torch
import torchvision.models as models


def main():
        model = load_model('pretrained/firenet_1000.pth.tar')
        device = get_device(use_gpu=True)
        model = model.to(device)
        model.eval()
        # TO DO

if __name__ == "__main__":
    main()