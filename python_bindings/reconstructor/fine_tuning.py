#!/usr/bin/env python3

from skimage.metrics import mean_squared_error, structural_similarity
from utils.loading_utils import load_model, get_device, load_raw_model
from torch.utils.data import DataLoader, Dataset
from model.model import *

from pathlib import Path
import numpy as np
import cv2

from fine_tuning_utils import EventDataset, ReconstructionLoss, arg_parser

def main(args):

    # Torch configuration: use GPU or CPU
    torch.cuda.empty_cache()
    device = get_device(use_gpu=True)

    # Initialize the dataset and the data loader
    event_dataset = EventDataset(
        args.dataset, 640, 480, args.sequence_len, args.n_sequences, args.n_noise_events)
    train_loader = DataLoader(event_dataset, batch_size=1, shuffle=True)

    # Load pretrained model and move it to the GPU
    model = load_model(args.pretrained_model)
    model = model.to(device).train()

    # Ensure that all the parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    # Define the optimizer and the loss criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = ReconstructionLoss(device)

    # Training metrics
    best_epoch = 0
    best_mean_ssim = -np.inf
    lpips_loss_epoch, ssim_epoch, mse_epoch = [], [], []

    # Create the metrics file with the header
    with open("fine_tuned/metrics.csv", "w") as f:
        f.write("epoch,lpips_loss,ssim,mse\n")

    # Training loop
    for e in range(args.epochs+1):

        # Iterate over the sequences random sampled
        for i, (x_seq, y_seq) in enumerate(train_loader):

            # Erase the hidden states at the beginning of each sequence
            states = None

            # Iterate over the sequence
            for ii in range(len(x_seq)):
                x = x_seq[ii]
                x = x.to(device)

                # Forward pass
                output, states = model(x, states)

                print(
                    f"Epoch [{e}/{args.epochs}], Sequence [{i+1}/{len(train_loader)}], Frame [{ii+1}/{len(x_seq)}] - Inference")

                # Show the output
                if args.show:
                    y_viz = y_seq[ii].squeeze().detach().numpy()
                    output_viz = output.squeeze().squeeze().cpu().detach().numpy()
                    cv2.imshow("Reconstructed vs Ground Truth", np.hstack(
                        (y_viz, output_viz)))
                    cv2.waitKey(1)

            # Compute the loss
            y = y_seq[-1].unsqueeze(1).to(device)
            loss = criterion.perceptual_loss(output, y)

            # Backpropagation and optimization
            if e > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Metics of the sequence
            lpips_loss = loss.item()
            mse = mean_squared_error(output.squeeze().squeeze().cpu().detach().numpy(),
                                     y.squeeze().squeeze().cpu().detach().numpy())

            ssim = structural_similarity(output.squeeze().squeeze().cpu().detach().numpy(),
                                         y.squeeze().squeeze().cpu().detach().numpy())

            if e == 0:
                print(
                    f"\033[33m\033[1m[INIT] Epoch [{e}/{args.epochs}], Sequence {i+1}/{len(train_loader)}], LPIPS loss: {lpips_loss}, SSIM: {ssim}, MSE: {mse}\033[0m")
            else:
                print(
                    f"\033[34m\033[1m[TRAIN]Epoch [{e}/{args.epochs}], Sequence {i+1}/{len(train_loader)}], LPIPS loss: {lpips_loss}, SSIM: {ssim}, MSE: {mse}\033[0m")

            lpips_loss_epoch.append(lpips_loss)
            ssim_epoch.append(ssim)
            mse_epoch.append(mse)

        # Metrics of the epoch
        mean_lpips_loss = np.mean(lpips_loss_epoch)
        mean_ssim = np.mean(ssim_epoch)
        mean_mse = np.mean(mse_epoch)
        print(
            f"\033[32mEpoch [{e+1}/{args.epochs}], Mean LPIPS loss: {mean_lpips_loss}, \
            Mean SSIM: {mean_ssim}, Mean MSE: {mean_mse}[0m")

        with open("fine_tuned/metrics.csv", "a") as f:
            f.write("{},{},{},{}\n".format(
                e+1, mean_lpips_loss, mean_ssim, mean_mse))

        if mean_ssim > best_mean_ssim:
            best_mean_ssim = mean_ssim
            best_epoch = e
            torch.save(model.state_dict(),
                       "fine_tuned/firenet_finetuned_best.pth")

            print("\033[32m\033[1m>>>[CHECKPOINT] Best model achived in epoch {}, new best SSIM: {}<<<\033[0m".format(
                best_epoch+1, best_mean_ssim))

    # Save the trained model
    raw_model = load_raw_model("pretrained/firenet_1000.pth.tar")
    raw_model["state_dict"] = model.state_dict()
    torch.save(raw_model, "fine_tuned/firenet_finetuned_full_raw.pth")
    torch.save(model.state_dict(), "fine_tuned/firenet_finetuned_full.pth")

if __name__ == "__main__":
    args = arg_parser()
    print("\033[1mFine tuning options:\033[0m")
    print(" - Model: {}".format(args.pretrained_model))
    print(" - Dataset: {}".format(args.dataset))
    print(" - Epochs: {}".format(args.epochs))
    print(" - Learning rate: {}".format(args.lr))
    print(" - Sequence Length: {}".format(args.sequence_len))
    print(" - Number of sequences: {}".format(args.n_sequences))
    print(" - Number of noise events: {}".format(args.n_noise_events))
    main(args)
