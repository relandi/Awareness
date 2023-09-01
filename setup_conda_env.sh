#!/bin/bash

sudo apt-get update

source ~/anaconda3/etc/profile.d/conda.sh

conda create -y -n awareness python=3.8 pip
conda activate awareness

pip install -e ./awareness/
pip install -e ./utils/
pip install git+https://github.com/openai/CLIP.git
pip install -r ./requirements.txt