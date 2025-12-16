#!/bin/bash
# Test the gradient_feeder_cpp extension
cd /mnt/c/dev/Project-ResNet-BK-An-O-N-Language-Model-Architecture
source venv_ubuntu/bin/activate
cd src/kernels
python -c "
import sys
import torch  # Import torch FIRST to load libc10.so
sys.path.insert(0, '.')
import gradient_feeder_cpp
print('SUCCESS: gradient_feeder_cpp loaded!')
print('Functions:', dir(gradient_feeder_cpp))
# Test FeederState
state = gradient_feeder_cpp.FeederState()
print(f'FeederState created, clip_threshold={state.clip_threshold}')
# Test feed function
threshold, scale, action = gradient_feeder_cpp.feed(state, 0.5)
print(f'feed(0.5): threshold={threshold}, scale={scale}, action={action}')
"
