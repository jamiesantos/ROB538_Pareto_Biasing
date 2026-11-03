#!/bin/bash

set -e

args=(
    ccea                              # Algorithm
    rover                             # Environment
    data                              # Training data
    config/ccea/4ag_2poi_ccea.yaml    # Algorithm config
    config/ccea/4ag_2poi_env.yaml     # Environment config
    0                                 # Random seed
    test                              # Run mode
    50                                # Iterations
)

python main.py "${args[@]}"