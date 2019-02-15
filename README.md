# TrackerNet
Object tracking using regression-based CNN

## Setup
Run the following two commands to replicate the Conda environment used for development and test and to obtain ?.

```
conda env
./init.sh
```

If one desires to do training, a local copy of [TrackingNet](http://www.lrgonzales.com/traffic-sign-classifier) is required.

## Training
`./init.sh` downloads frame groupings


## Inference
Inference requires a JSON to structure the adjacent frames on which to perform inference. Refer to `data/inference_dog.json` or `data/inference_kid.json` for an example. Given a properly formatted input JSON file, run inference with `python src/trackernet_inference.py <model_file> <json_file>`, where `<model_file>` is a `.h5` saved Keras model and `<json_file>` is a properly formatted `.json` file.
