# rooftop_data
This package downloads and creates yaml files (to be used as [PolygonWithHoles](https://github.com/ethz-asl/mav_comm/blob/master/mav_planning_msgs/msg/PolygonWithHoles.msg) ROS messages) from the [EPFL polygon rooftop](https://cvlab.epfl.ch/data/data-polygonalobjectdataset/) dataset.

## Requirements
```
pip install numpy scipy imageio argparse matplotlib pyyaml urllib3 pyyaml
```

## Build dataset
This will download the dataset, and create the yaml files in ./data/Rooftop/yaml
```
python read_rooftop_data.py -p
```
(The `-p` flag will also plot a random image from the dataset)
