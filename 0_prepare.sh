#!/bin/bash
echo 'Activating ml environment ...' 
eval "$(conda shell.bash hook)"
conda activate ml &&
conda info | grep "active environment" &&

echo 'Entering GPU node ...'
srun -p gpushort --gres=gpu:4 --constraint=pascal  --time=4:00:00 --mem=6000M --pty bash
