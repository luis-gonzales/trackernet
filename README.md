# TrackerNet
Object tracking using regression-based CNN. The work is motivated by .

## Setup
Run the following two commands to replicate the Conda environment used for development and test and to obtain ?.

```
conda env
./init.sh
```

If one desires to do training, a local copy of [TrackingNet](http://www.lrgonzales.com/traffic-sign-classifier) is required.

## Model Architecture

## Training
`./init.sh` downloads frame groupings


## Inference
Inference on adjacent frames can be performed using `python src/trackernet_inference.py <model_file> <json_file>`. `<model_file>` is a `.h5` saved Keras model, while `<json_file>` is a JSON file that contains paths to two images and a bounding box for the object to track from the first image. See `data/inference_dog.json` and `data/inference_kid.json` for examples of a properly structured JSON file.

## Improvements
Given the exploratory and time-constrained nature of this project, fine-tuning of hyperparameters and model architecture is pending. The "head" of the CNN architecture is probably okay with being the first several layers of a pretrained model. However, the "tail" of the CNN is . In addition, hyperparameter tuning could lead to improved performance.

Padding for detection box

## References
[1] [Learning to Track at 100 FPS with Deep Regression Networks, D. Held et al., 2016](https://arxiv.org/pdf/1604.01802.pdf)

