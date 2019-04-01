#!/bin/bash

for (( i=0; i<101; i++ )) ; do
  python3 generate_synthetic_data.py -ns 100 -no ${i} -d ./data/${i}
  mkdir -p ./export/${i}
  cp ./data/${i}/*.yaml ./export/${i}
done
