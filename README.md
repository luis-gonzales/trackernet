# TrackerNet
TrackerNet performs object tracking using a regression-based convolutional neural network. Given two input frames, the CNN strives to find the desired object from the "previous" frame in the "current" frame. The custom CNN architecture is inspired by GOTURN object tracking [1] and YOLO object detection [2].


## Setup
The Conda environment used for development and test can be replicated by running `conda env create -f trackernet.yml`.

If one desires to perform training, the following are required:

1. YOLO weights. These are used to initialize the "head" of the CNN architecture (more on this in the Overview section).
2. [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit) dataset. The entire dataset is not required. In fact, TrackerNet currently only utilizes `TRAIN_0`, `TRAIN_1`, `TRAIN_2`, and `TRAIN_3` for its dataset.
3. Because the CNN requires two inputs, a file is needed to associate adjacent frames. This was accomplished using a JSON file 

Running `./init.sh` takes care of  requirements 1 and 3 above.


## Overview
Below is a conceptual depiction of TrackerNet.
Config file


## Model Architecture



## Training
`./init.sh` downloads frame groupings


## Inference
Inference on adjacent frames can be performed using `python src/trackernet_inference.py <model_file> <json_file>`. `<model_file>` is a `.h5` saved Keras model, while `<json_file>` is a JSON file that contains paths to two images and a bounding box for the object to track from the first image. See `data/inference_dog.json` and `data/inference_kid.json` for examples of a properly structured JSON file.

## Improvements
Given the exploratory and time-constrained nature of this project, fine-tuning of hyperparameters and model architecture is pending. The "head" of the CNN architecture is probably okay with being the first several layers of a pretrained model. However, the "tail" of the CNN is . In addition, hyperparameter tuning could lead to improved performance.

Padding for detection box

Anchor boxes

## References
[1] [Learning to Track at 100 FPS with Deep Regression Networks, D. Held et al., 2016](https://arxiv.org/pdf/1604.01802.pdf)
[2] [YOLOv3: An Incremental Improvement, J. Redmon et al., 2018](https://arxiv.org/pdf/1804.02767.pdf)

