#!/usr/bin/bash

mamba create --name humanPlayV2 python=3.8 -y
mamba activate humanPlayV2

pip install gymnasium[atari]
pip install gymnasium[accept-rom-license] 
AutoROM --install-dir ROMS -y
pip install imageio
pip install stable-baselines3
pip install opencv-python
mamba install moviepy ocl-icd-system -y


# pip install gymnasium[classic-control]

# pip install autorom[accept-rom-license] 

# pip install IPython
# pip install --upgrade gym[atari] ale_py pygame
# AutoROM --accept-license

# # pip install gymnasium[other]

# sudo apt install gstreamer1.0-plugins-bad -y

python play/saveGifPlay.py