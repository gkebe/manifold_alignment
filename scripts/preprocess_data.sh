#!/bin/bash
python split_data_pre_loader.py --data=data/gld_$1_vision_tensors.pkl --train=data/gld_$1_vision_train.pkl --test=data/gld_$1_vision_test.pkl --dev=data/gld_$1_vision_dev.pkl
python generate_negatives.py --data_file=data/gld_$1_vision_train.pkl --out_file=data/gld_$1_vision_train_pos_neg.pkl
python generate_negatives.py --data_file=data/gld_$1_vision_dev.pkl --out_file=data/gld_$1_vision_dev_pos_neg.pkl
python generate_negatives.py --data_file=data/gld_$1_vision_test.pkl --out_file=data/gld_$1_vision_test_pos_neg.pkl
