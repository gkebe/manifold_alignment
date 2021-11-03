#!/bin/bash
cd "$(dirname "$0")"
cd ..
for i in $(seq 1 5); 
do 
python plotting/create_epochs_plots.py --experiment ${1}_${i} --threshold ${2} --dev
mv output/${1}_${i}/${1}_${i}_epochs.pkl plots/
done
