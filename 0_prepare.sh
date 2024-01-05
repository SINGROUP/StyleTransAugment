#!/bin/bash
echo 'Activating ml environment ...' 
eval "$(conda shell.bash hook)"
conda activate ml &&
conda info | grep "active environment" &&

# If the host is a GPU node (node name contain gpu or dgx), then do nothing, else enter a GPU node
if [[ $HOSTNAME == *"gpu"* ]] || [[ $HOSTNAME == *"dgx"* ]]; then
    echo "Already in GPU node $HOSTNAME"
else
    echo "Entering GPU node ..."
    srun -p gpushort --gres=gpu:4 --constraint=pascal  --time=4:00:00 --mem=6000M --pty bash
fi
