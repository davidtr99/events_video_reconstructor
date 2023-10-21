import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from utils.inference_utils import events_to_voxel_grid
from pathlib import Path
import numpy as np
import os
import cv2
import argparse

### EVENT DATASET ###
# Dataset is in the form of a sequence of events and images
# img0.png , ev0.csv, img1.png, etc.
# ev0.csv contains the events from img0.png to img1.png and so on


class EventDataset(Dataset):
    def __init__(self, path, width, height, sequence_len=10, n_sequences=10,
                 n_noise_events=0, img_ext=".png", ev_ext=".csv"):
        """
        Args:
            path (string): Path to the folder containing the dataset
            width (int): Width of the images
            height (int): Height of the images
            sequence_len (int): Length of each generated sequence
            n_sequences (int): Number of sequences to generate
            n_noise_events (int): Number of noise events to add to each tensor
            img_ext (string): Extension of the images
            ev_ext (string): Extension of the events
        """

        # Get the filenames of the events and images
        trainpath = Path(path)
        filenames = [item for item in os.scandir(trainpath) if item.is_file()]
        events_filenames = [Path(item).as_posix()
                            for item in filenames if Path(item).suffix == ev_ext]
        images_filenames = [Path(item).as_posix()
                            for item in filenames if Path(item).suffix == img_ext]
        events_filenames.sort()
        images_filenames.sort()

        # Removing first image filename (no events previously)
        images_filenames = images_filenames[1:]

        # Check if number of events files and images match
        assert len(events_filenames) == len(images_filenames)

        # Generate random window samplings of the full sequence dataset
        # (will generate n_sequences sequences of length sequence_len)
        self.init_indexes = []
        self.end_indexes = []
        for _ in range(n_sequences):
            init_index = np.random.randint(
                0, len(events_filenames) - sequence_len)
            end_index = init_index + sequence_len
            self.init_indexes.append(init_index)
            self.end_indexes.append(end_index)

        # Store parameters
        self.width = width
        self.height = height
        self.sequence_len = sequence_len
        self.n_sequences = n_sequences
        self.n_noise_events = n_noise_events
        self.events_filenames = events_filenames
        self.images_filenames = images_filenames

    def __getitem__(self, sq):
        """
        Args:
            sq (int): Sequence index to retrieve
        Returns:
            events_tensor_sequence (list): List of event tensors
            image_sequence (list): List of images
        """

        # Initialize sequences
        events_tensor_sequence = []
        image_sequence = []

        for i in range(self.init_indexes[sq], self.end_indexes[sq]):

            # Reading data from files
            image_filename = self.images_filenames[i]
            events_filename = self.events_filenames[i]
            image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE) / 255.0
            image = torch.from_numpy(image).float()
            events = np.loadtxt(events_filename, delimiter=",",
                                skiprows=1, usecols=(1, 2, 3, 4))

            # Add noise to the event tensor if required
            if self.n_noise_events > 0:

                # Extracting relevant data
                w = self.width
                h = self.height
                n = self.n_noise_events
                t_init = events[0, 1]
                t_end = events[-1, 1]

                # Generating noise event tensor
                noise_events = np.zeros((self.n_noise_events, 4))
                noise_events[:, 0] = np.random.choice([-1, 1], n)
                noise_events[:, 1] = np.random.randint(t_init, t_end, n)
                noise_events[:, 2] = np.random.randint(0, w, n)
                noise_events[:, 3] = np.random.randint(0, h, n)

                # Concatenate noise events to the original tensor
                # and sort temporally
                events = np.concatenate((events, noise_events), axis=0)
                events = events[events[:, 1].argsort()]

            # Generate the window in the required format
            event_window = np.zeros((4, len(events)))
            event_window[0, :] = events[:, 1]  # ts
            event_window[1, :] = events[:, 2]  # x
            event_window[2, :] = events[:, 3]  # y
            event_window[3, :] = events[:, 0]  # p

            # Voting (voxel grid)
            events_voxel_grid = events_to_voxel_grid(
                event_window.T, num_bins=5, width=self.width, height=self.height)
            events_voxel_grid = torch.from_numpy(events_voxel_grid).float()

            # Add to the required sequence
            events_tensor_sequence.append(events_voxel_grid)
            image_sequence.append(image)

        return events_tensor_sequence, image_sequence

    def __len__(self):
        """
        Returns:
            n_sequences (int): Number of sequences in the generated data
        """
        return self.n_sequences

### PERCEPTUAL MEASURE OF SIMILARITY ###
# Based on  VGG-19 (NN model useful to measure the perceptual
# similarity) and used to compute the LPIPS loss


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            weights="VGG19_Weights.DEFAULT"
        ).features
        vgg_pretrained_features[0] = torch.nn.Conv2d(
            1, 64, kernel_size=3, padding=1)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

### RECONSTRUCTION LOSS ###
# Based on the LPIPS loss (perceptual similarity) and the MSE loss
# (pixel-wise similarity)


class ReconstructionLoss:
    def __init__(self, device):
        self.vgg_module = VGG19().to(device).eval()
        self.lpips_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.mse_module = torch.nn.MSELoss()

    def perceptual_loss(self, pred, y):
        pred_vgg, y_vgg = self.vgg_module(pred), self.vgg_module(y)
        lpips_loss = 0
        for i in range(len(pred_vgg)):
            lpips_loss += self.lpips_weights[i] * \
                torch.nn.functional.l1_loss(pred_vgg[i], y_vgg[i])
        return lpips_loss

### PARSER CONFIGURATION ###
# Parse the arguments from the command line


def arg_parser():
    parser = argparse.ArgumentParser(description="Fine tuning")
    parser.add_argument("--pretrained_model", type=str,
                        help="Model to fine tuning", required=True)
    parser.add_argument("--dataset", type=str,
                        help="Dataset to fine tuning", required=True)
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs", default=1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument("--sequence_len", type=int,
                        help="Sequence length", default=10)
    parser.add_argument("--n_sequences", type=int,
                        help="Number of sequences", default=50)
    parser.add_argument("--n_noise_events", type=int,
                        help="Number of noise events", default=0)
    parser.add_argument("--show", type=bool,
                        help="Show images", default=False)
    args = parser.parse_args()
    return args
