#!/bin/bash

for (( i=0; i<101; i++ )) ; do
  yaml_dir="./export/${i}"
  mkdir -p $yaml_dir
  python3 generate_synthetic_data.py -ns 100 -no ${i} -d ./data/ -y $yaml_dir
done
