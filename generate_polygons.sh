#!/bin/bash

for (( i=0; i<16; i+=5 )) ; do
  yaml_dir="./export/${i}"
  mkdir -p $yaml_dir
  python3 generate_synthetic_data.py -ns 100 -no ${i} -d ./data/ -y $yaml_dir --hull_size 1000 1000 --no_overlaps --scale 0.1
done
