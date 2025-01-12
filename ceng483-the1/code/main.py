
from experiment import *
import yaml

with open('experiment_configurations.yaml', 'r') as file1:
    sections = yaml.safe_load(file1)
with open('results.txt', 'w') as file2:
    for section in sections.keys():
        exps=sections[section]
        for exp in exps.keys():
            config=exps[exp]
            accuracy=instance_recognition(config['query_number'],config['grid_n'],config['color_histogram_type'],config['color_space_type'],config['quantization_interval'])
            file2.write("configurations: " + str(config) + "\n")
            file2.write("accuracy: " + str(accuracy) + "\n" + "\n")


