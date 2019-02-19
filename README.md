# TrackerNet
TrackerNet performs object tracking using a regression-based convolutional neural network. Given two input frames, the CNN strives to find the desired object from the "previous" frame in the "current" frame. The custom CNN architecture is inspired by GOTURN object tracking [1] and YOLO object detection [2].

## Setup
The Conda environment used for development and test can be obtained by running `conda env create -f <yml_file>` where `<yml_file>` is either `trackernet.yml` or `trackernet-gpu.yml`.

If one desires to perform training, the following are required:  Instead, describe what `./init.sh` downloads (change due to model having to go on Google Drive.

1. YOLO weights. These are used to initialize the parameters in the "head" of the CNN architecture (more on this in the Overview section).
2. [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit) dataset. The entire dataset is not required. In fact, TrackerNet is currently trained on a subset of the dataset, including TRAIN_0, TRAIN_1, TRAIN_2, and TRAIN_3. Describe skip every 5
3. Because the CNN requires two input images per forward pass, a file is needed to manage the associations. A JSON file was used to group adjacent frames of TrackingNet.

Running `./init.sh` downloads (a) open-source YOLO parameters and (b) the specific dataset used during development to `data/` (requirements 1 and 3 above).

Alternatively, 

## Overview
Below is a conceptual depiction of TrackerNet.
Config file

## Model Architecture
Description here.

## Training
As mentioned in the Setup section, `./init.sh` downloads `data_train.json`, `data_val.json`, and `data_test.json` to `data/`. Because a user may not require all of TrackingNet's TRAIN_0, ... TRAIN_3 during their training, `src/trackingnet_local.py` is used to create JSONs that correspond to the user's local copy of TrackingNet (e.g., `data_train_local.json`).

Training can then be performed by running `python src/trackernet_train <cfg_head> <cfg_tail>` where `<cfg_head>` and `<cfg_tail>` are text files (refer to `cfg/trackernet_head.cfg` and `cfg/trackernet_tail.cfg`). Note that `<cfg_head>` is expected to contain the model hyperparameters. Checkpoints are saved to the `model/checkpoints` directory.

## Inference
Inference can be performed by running `python src/trackernet_inference.py <model_file> <json_file>`. `<model_file>` is expected to be a `.h5` saved Keras model, and `<json_file>` must be a JSON file that contains paths to two images and a bounding box for the object to track from the first image. See `data/inference_elephant_*.json` and `data/inference_skater.json` for examples of the required formatting.

## Improvements
Given the exploratory and time-constrained nature of this project, fine-tuning of hyperparameters and the model architecture is pending. As far as the CNN "head", it's not immediately clear what spatial resolution is optimal before the concatenation operation. The current model uses a 1-by-1 kernel immediately after concatenation to compress the depth of the resultant tensor; this could prove to be a bottleneck. Similarly, the specific operations comprising the "tail" of the CNN need to be optimized.

TrackingNet is made up of a wide range of objects and corresponding geometries, including humans, dogs, canoes, elephants, and more. When faced with varying object geometries, recent object detectors have resorted to using appropriately-sized anchor boxes. Correspondingly, it's expected that TrackerNet performance would be improved with appropriately-sized anchors. Presently, TrackerNet simply uses a fixed 96-by-96 anchor box.

The training data for the current model was created by grouping every fifth (every ~0.17 sec) from of TrackingNet. The motivation was that this . Improvements may be attainable by curating a dataset that specifically includes occlusions, lighting changes, etc         training data

Finally, when cropping the "previous" image, it may be beneficial to include slightly more pixels than the crop obtained by directly using the corresponding bounding box. Doing so may mitigate any loss of information caused by the convolution operations along the borders.

## Customization
TrackerNet was designed such that the "current" image input has a field of view that is twice that of the "previous" image input.    FOV mult

## References
[1] [Learning to Track at 100 FPS with Deep Regression Networks, D. Held et al., 2016](https://arxiv.org/pdf/1604.01802.pdf)

[2] [YOLOv3: An Incremental Improvement, J. Redmon et al., 2018](https://arxiv.org/pdf/1804.02767.pdf)

