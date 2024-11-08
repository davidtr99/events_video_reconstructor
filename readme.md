# Events Video Reconstructor
This package implements a ROS node which subscribes to an event topic and reconstructs the video from the events. The ROS node is implemented in C++ and the network inference is developed in Python using PyTorch. It also has utils to fine tune the network with a custom dataset and to detect the hot pixels noisy events in the event stream.

## Dependencies
* ROS Noetic
* OpenCV 
* PyTorch
* skimage
* PyBind11

## Usage
### Reconstruct video from events
To reconstruct the video from the events, you need to run the following command:
```
    roslaunch events_video_reconstructor node.launch 
```
The launchfile has the following arguments:
* `event_topic`: The topic where the events are published. Default: `/dvs/events`
* `video_topic`: The topic where the reconstructed image is published. Default: `/dvs/reconstructed_image`
* `output_frequency`: The frequency of the reconstructed image. A negative value will generate a reconstructed image per arrived event package.  Default: `-1`

The internal parameters of the network can be modified in the [config/inference_params.yaml](config/inference_params.yaml) file. These parameters control the reconstruction process, selecting the network weights file, postprocessing parameters, etc.

### Fine tune the network
To fine tune the network with a custom dataset, you need to run the following command with this example configuration:
```
    python3 fine_tuning.py --pretrained_model pretrained/firenet_1000.pth.tar --dataset data/train/testbed_shelves_dataset --epochs 15 --sequence_len 10 --n_sequences 20 --lr 1e-7 --n_noise_events 10
```

The dataset argument must point to a folder with CSV files containing the events, and png images with the ground truth. The CSV files must have the following format:
```
    index, polarity, timestamp, x, y
```

You can see a description of the arguments running:
```
    cd python_bindings/reconstructor/
    python3 fine_tuning.py --help
```

### Detect hot pixels
To detect the hot pixels in the event stream, you can simply run:
```
    rosrun events_video_reconstructor hot_pixels_detector.py
```
Please, point to a static scene and don't move the camera during this process. A hot_pixels.txt file will be generated in the correct location to be used by the network.

### Network architecture
The network architecture is FireNet. The inner code of the model is forked of [FireNet Repository](https://github.com/cedric-scheerlinck/rpg_e2vid/tree/cedric/firenet).

## Troubleshooting
Contact: David Tejero-Ruiz (dtejero@catec.aero)

Found a bug? Create an ISSUE!

Do you want to contribute? Create a PULL-REQUEST!