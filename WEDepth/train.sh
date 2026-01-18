#!/bin/bash

python3 -m torch.distributed.launch --nproc_per_node=4 train.py --config-file "./configs/nyu.json"