#!/usr/bin/bash

mamba create --name humanPlay python=3.8 gymnasium[atari,classic-control,accept-rom-license] -y
mamba activate humanPlay

pip install gymnasium[atari]
pip install gymnasium[classic-control]
pip install gymnasium[accept-rom-license] 
pip install autorom[accept-rom-license] 
pip install imageio
pip install IPython
pip install opencv-python
AutoROM --install-dir ROMS -y


pip install --upgrade gym[atari] ale_py pygame
AutoROM --accept-license
pip install stable-baselines3
# pip install gymnasium[other]
mamba install moviepy ocl-icd-system -y
sudo apt install gstreamer1.0-plugins-bad -y

python play/saveGifPlay.py