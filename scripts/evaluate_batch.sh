#!/bin/bash
cd "$(dirname "$0")"
cd ..
for i in $(seq 1 5); 
do
echo ${1}_${i}
python evaluate.py --experiment_name ${1}_${i}_bias --test_data_path ./data/gld_${1}_vision_test.pkl --pos_neg_examples_file ./data/gld_${1}_vision_test_pos_neg.pkl --threshold ${2} --epoch ${3}
#mv output/${1}_${i}/${1}_${i}_epochs.pkl plots/
done
