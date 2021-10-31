#!/bin/bash
cd "$(dirname "$0")"
cd ..
model=${1}
epochs=${3}
initial_seed=${4}
gpu=${5}
for i in {1..${2}}
do
   seed=$((initial_seed + i))
   python train.py --experiment_name=${model}_${i} --epochs=${epochs} --seed=${seed} --train_data=data/gld_${model}_vision_train.pkl --pos_neg_examples_file=data/gld_${model}_vision_train_pos_neg.pkl --dev_data=data/gld_${model}_vision_dev.pkl --test_data=data/gld_${model}_vision_test.pkl --gpu_num=${gpu}
done