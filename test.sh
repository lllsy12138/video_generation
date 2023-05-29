#!/bin/bash
yhrun -N 1 -p gpu --gpus-per-node=1 --cpus-per-gpu=8 python test.py