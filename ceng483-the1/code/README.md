# Instance Recognition with Color Histograms

In this project, we are utilizing color histograms with the goal of instance recognition on datasets with different characteristics. 
Various configurations are used during the creation of color histograms. 

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Files](#files)


## Features

- 3D and Per-channel histogram implementations
- RGB color space to HSV color space conversion
- Grid based feature extraction
- Image similarity computation

## Getting Started

### Prerequisites

- The python libraries OpenCV, PyYAML, OS and Numpy has to be installed beforehand.
- The dataset folder has to be in the same folder with the provided python files.

## Usage
- The experiment_configurations.yaml file includes all the configurations I have used 
 throughout the experiment and in the report. All these configurations can be tried by only 
 running the main.py file.
- If a different configuration that is not included in the yaml file wants to be used,
either yaml file can be changed in the format that is provided in it (includes a section that includes experiment which includes the parameters for the instance recognition function) 
or main.py file can be changed. 
- If changing the main.py file is prefered, the specific configuration can be tried as
instance_recognition(query_number, n_by_n, color_histogram_type, color_space_type, quantization_interval)
- Running the main.py file will write the results into the results.txt file.

## Files
- main.py: importes the functions from experiment.py, parses the yaml file that includes the configurations
and writes the results into results.txt
- experiment.py: includes the instance_recognition function that accepts the paramaters: \
query_number: an integer representing which query set you are running the experiment on; can be 0,1 or 2. \
n_by_n: an integer representing the grid number, if n is 2 grid size is 2x2. \
color_histogram_type: a string representing the color histogram type used, can be "per-channel" or "3d" \
color_space_type: a string representing the color space type, can be "rgb" or "hsv" \
quantization_interval: an integer representing the quantization interval for color histograms
- color_histogram.py: includes the implementations of: \
per_channel_color_histogram \
threed_color_histogram \
rgb_to_hsv_color_space_calculation \
grid_based_feature_extraction 
- experiment_configurations.yaml: includes all the configurations that is tried on report.
- results.txt: results are written into this file when main.py file is run.