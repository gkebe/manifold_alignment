#!/bin/bash

for i in $(seq 1 5); 
do
echo ${1}_${i}
echo ${3}
python evaluate_cosine_test.py --experiment_name ${1}_${i} --test_data_path ./data/gld_${1}_vision_test_${3}.pkl --pos_neg_examples_file ./data/gld_${1}_vision_test_${3}_pos_neg.pkl --threshold ${2} --group ${3}
#mv output/${1}_${i}/${1}_${i}_epochs.pkl plots/
done
