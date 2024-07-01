# sudo apt-get install zlib1g-dev -y
mamba env create -f environment.yml
mamba activate kstu_model_education_analisys_tool

AutoROM --install-dir ROMS -y

mamba install atari_py
python -m atari_py.import_roms ROMS

ln -s /home/fdobrotv/anaconda3/lib/libopenh264.so.5 /home/fdobrotv/miniforge3/envs/kstu_model_education_analisys_tool/lib/libopenh264.so.5