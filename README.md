# rooftop_data
This package downloads and creates yaml messages for the [EPFL polygon rooftop](https://cvlab.epfl.ch/data/data-polygonalobjectdataset/) dataset.

## Requirements
'''
pip install numpy scipy imageio argparse matplotlib pyyaml
'''

## Build dataset
This will download the dataset, and create the yaml files in ./data/Rooftop/yaml
'''
python read_rooftop_data -p
'''
(The '-p' flag will also plot a random image from the dataset)