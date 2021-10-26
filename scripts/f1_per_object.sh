#!/bin/bash
cd ..
#for i in $(seq 1 5); 
#do
#echo ${1}_${i}
python evaluate_per_object.py --experiment_name ${1} --threshold ${2} --epoch ${3}
#mv output/${1}_${i}/${1}_${i}_epochs.pkl plots/
#done
