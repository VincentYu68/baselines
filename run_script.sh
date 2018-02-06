#!/bin/bash

mpirun -np 4 python -m baselines.split_net.run_exp --seed 1
mpirun -np 4 python -m baselines.split_net.run_exp --seed 2
mpirun -np 4 python -m baselines.split_net.run_exp --seed 3