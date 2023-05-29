#!/bin/bash
#yhrun -N 1 -p debug python main.py
#yhrun -N 1 -p gpu --gpus=2 --gpus-per-node=2 --cpus-per-gpu=8 python main.py
yhrun -N 1 -p gpu --gpus-per-node=1 --cpus-per-gpu=8 python main.py