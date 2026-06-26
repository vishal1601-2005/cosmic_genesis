#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
source ~/cosmic_gpu/bin/activate
python main.py "$@"
