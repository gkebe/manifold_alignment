#!/bin/bash
cd "$(dirname "$0")"
cd ..
model=${1}
run=${2}
epochs=${3}
seed=${4}
python train.py --experiment_name=${model}_${run} --epochs=${epochs} --seed=${seed} --train_data=data/gld_${model}_vision_train.pkl --pos_neg_examples_file=data/gld_${model}_vision_train_pos_neg.pkl --dev_data=data/gld_${model}_vision_dev.pkl --test_data=data/gld_${model}_vision_test.pkl --gpu_num=0