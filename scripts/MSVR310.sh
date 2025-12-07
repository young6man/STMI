#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate STMI
cd /vepfs-cnbj3fa964354bf4/xuxg/reid/STMI
python train.py --config_file /vepfs-cnbj3fa964354bf4/xuxg/reid/STMI/configs/MSVR310/STMI.yml
