#!/usr/bin/bash

mamba create --name wandbEnv python=3.8 wandb -y
mamba activate wandbEnv

docker run --rm -d -v wandb:/vol -p 8080:8080 --name wandb-local wandb/local