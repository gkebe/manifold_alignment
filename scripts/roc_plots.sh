#!/bin/bash
models=( "vq-wav2vec" "decoar" "wav2vec2" "transcriptions" "w2v2transcriptions" "mfcc")
cd ..
for i in {0..5} ;
do
    model=${models[${i}]}
    for j in {1..5} ;
    do
        python ./plotting/create_auc_plots_dev.py --experiment ${model}_${j}
        mv output/${model}_${j}/${model}_${j}_auc.pkl plots/
    done
done
