#!/bin/sh

python -m pip install --upgrade pip
pip install torch==1.4.0 --quiet  --find-links https://download.pytorch.org/whl/cpu/torch_stable.html --upgrade
pip install -r requirements.txt
pip install git+https://github.com/eleurent/highway-env
