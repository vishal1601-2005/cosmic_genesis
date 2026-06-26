#!/bin/bash
export PYTHONPATH=$(pwd):$(pwd)/physics:$(pwd)/epochs:$PYTHONPATH
source ~/cosmic_gpu/bin/activate

python -c "
import sys
sys.path.insert(0, '$(pwd)')
sys.path.insert(0, '$(pwd)/physics')
sys.path.insert(0, '$(pwd)/epochs')
print('✅ Python paths fixed')
"

python main.py "$@"
