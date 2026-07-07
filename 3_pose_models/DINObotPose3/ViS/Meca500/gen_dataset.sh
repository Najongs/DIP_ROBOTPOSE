#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dino
export CUDA_VISIBLE_DEVICES=GPU-05b804ff-3b02-39f4-cf62-b848e189ebdd
OUT=/home/najo/NAS/DIP/datasets/meca500_synth
python3 ViS/Meca500/synth_gen.py --n 30000 --seed 0   --out $OUT/train > ViS/Meca500/gen_train.log 2>&1
python3 ViS/Meca500/synth_gen.py --n 3000  --seed 999 --out $OUT/val   > ViS/Meca500/gen_val.log 2>&1
echo GEN_DONE > ViS/Meca500/GEN_DONE
