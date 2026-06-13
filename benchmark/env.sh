#!/bin/bash
# Load the toolchain modules for building/running the airway pipeline.
# Modules must be loaded sequentially (cuda first), per the cluster's Lmod setup.
module load cuda/12.8
module load openmpi/5.0.2_ucx1.14.1_cuda12.3
