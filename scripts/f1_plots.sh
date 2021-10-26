#!/bin/bash
cd ..
for i in $(seq 1 5); 
do 
python plotting/create_f1_plots_dev.py --experiment ${1}_${i} --epoch ${2} --dev
mv output/${1}_${i}/${1}_${i}_p_r_f1.pkl plots/ 
done
