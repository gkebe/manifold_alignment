#!/bin/bash

for i in $(seq 1 5); 
do 
python plotting/create_epochs_plots.py --experiment ${1}_${i} --threshold ${2}
#mv output/${1}_${i}/${1}_${i}_epochs.pkl plots/ 
done
