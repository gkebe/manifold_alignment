#!/bin/bash
cd "$(dirname "$0")"
cd ..

ACCENT_USER_IDS="$(python ./gold_featurization/speakers.py --trait accent --val yes)"
NONACCENT_USER_IDS="$(python ./gold_featurization/speakers.py --trait accent --val no)"

python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train_limit=2984 --train=data/groups/gld_${1}_accent_vision_train.pkl --test=data/groups/gld_${1}_accent_vision_test.pkl --test_limit=785 --users "$ACCENT_USER_IDS" >> data/groups/accent.txt

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_accent_vision_train.pkl --out_file=data/groups/gld_${1}_accent_vision_train_pos_neg.pkl

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_accent_vision_test.pkl --out_file=data/groups/gld_${1}_accent_vision_test_pos_neg.pkl


python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl  --train_limit=2984 --train=data/groups/gld_${1}_nonaccent_vision_train.pkl --test=data/groups/gld_${1}_nonaccent_vision_test.pkl --test_limit=785 --users "$NONACCENT_USER_IDS" >> data/groups/nonaccent.txt


MEN_USER_IDS="$(python ./gold_featurization/speakers.py --trait gender --val man)"

WOMEN_USER_IDS="$(python ./gold_featurization/speakers.py --trait gender --val woman)"

python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_men_vision_train.pkl --train_limit 3584 --test_limit 949 --test=data/groups/gld_${1}_men_vision_test.pkl --users $MEN_USER_IDS >> data/groups/men.txt

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_men_vision_train.pkl --out_file=data/groups/gld_${1}_men_vision_train_pos_neg.pkl

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_men_vision_test.pkl --out_file=data/groups/gld_${1}_men_vision_test_pos_neg.pkl


python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_women_vision_train.pkl --train_limit 3584 --test_limit 949 --test=data/groups/gld_${1}_women_vision_test.pkl --users $WOMEN_USER_IDS >> data/groups/women.txt

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_women_vision_train.pkl --out_file=data/groups/gld_${1}_women_vision_train_pos_neg.pkl

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_women_vision_test.pkl --out_file=data/groups/gld_${1}_women_vision_test_pos_neg.pkl



MUFFLED_USER_IDS="$(python ./gold_featurization/speakers.py --trait muffled-ness --val 1,2)"

NONMUFFLED_USER_IDS="$(python ./gold_featurization/speakers.py --trait muffled-ness --val 3)"

python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_muffled_vision_train.pkl --train_limit=1596 --test_limit=438 --test=data/groups/gld_${1}_muffled_vision_test.pkl --users $MUFFLED_USER_IDS >> data/groups/muffled.txt

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_muffled_vision_train.pkl --out_file=data/groups/gld_${1}_muffled_vision_train_pos_neg.pkl

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_muffled_vision_test.pkl --out_file=data/groups/gld_${1}_muffled_vision_test_pos_neg.pkl


python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_nonmuffled_vision_train.pkl --train_limit=1596 --test_limit=438 --test=data/groups/gld_${1}_nonmuffled_vision_test.pkl --users $NONMUFFLED_USER_IDS >> data/groups/nonmuffled.txt

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_nonmuffled_vision_train.pkl --out_file=data/groups/gld_${1}_nonmuffled_vision_train_pos_neg.pkl

python ./preprocessing/generate_negatives.py --data_file=data/groups/gld_${1}_nonmuffled_vision_test.pkl --out_file=data/groups/gld_${1}_nonmuffled_vision_test_pos_neg.pkl



BACKGROUND_USER_IDS="$(python ./gold_featurization/speakers.py --trait background_noise --val 1,2)"

NOBACKGROUND_USER_IDS="$(python ./gold_featurization/speakers.py --trait background_noise--val 3,4)"

python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_background_vision_train.pkl --train_limit=720 --test_limit=215 --test=data/groups/gld_${1}_background_vision_test.pkl --users $BACKGROUND_USER_IDS >> data/groups/background.txt

python generate_negatives.py --data_file=data/groups/gld_${1}_background_vision_train.pkl --out_file=data/groups/gld_${1}_background_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/groups/gld_${1}_background_vision_test.pkl --out_file=data/groups/gld_${1}_background_vision_test_pos_neg.pkl


python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_nobackground_vision_train.pkl --train_limit=720 --test_limit=215 --test=data/groups/gld_${1}_nobackground_vision_test.pkl --users $NOBACKGROUND_USER_IDS >> data/groups/nobackground.txt

python generate_negatives.py --data_file=data/groups/gld_${1}_nobackground_vision_train.pkl --out_file=data/groups/gld_${1}_nobackground_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/groups/gld_${1}_nobackground_vision_test.pkl --out_file=data/groups/gld_${1}_nobackground_vision_test_pos_neg.pkl



HIGHVOLUME_USER_IDS="$(python ./gold_featurization/speakers.py --trait volume --val 1)"

MEDIUMVOLUME_USER_IDS="$(python ./gold_featurization/speakers.py --trait volume --val 2,3)"

LOWVOLUME_USER_IDS="$(python ./gold_featurization/speakers.py --trait volume --val 4)"

python ./preprocessing/split_data_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_highvolume_vision_train.pkl --train_limit=316 --test_limit=109 --test=data/groups/gld_${1}_highvolume_vision_test.pkl --users $HIGHVOLUME_USER_IDS >> data/groups/highvolume.txt

python generate_negatives.py --data_file=data/groups/gld_${1}_highvolume_vision_train.pkl --out_file=data/groups/gld_${1}_highvolume_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/groups/gld_${1}_highvolume_vision_test.pkl --out_file=data/groups/gld_${1}_highvolume_vision_test_pos_neg.pkl


python split_data_pre_loader_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_mediumvolume_vision_train.pkl --train_limit=316 --test_limit=109 --test=data/groups/gld_${1}_lowvolume_vision_test.pkl --users $LOWVOLUME_USER_IDS >> data/groups/lowvolume.txt

python generate_negatives.py --data_file=data/groups/gld_${1}_lowvolume_vision_train.pkl --out_file=data/groups/gld_${1}_lowvolume_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/groups/gld_${1}_lowvolume_vision_test.pkl --out_file=data/groups/gld_${1}_lowvolume_vision_test_pos_neg.pkl


python split_data_pre_loader_group.py --data=data/gld_${1}_vision_tensors.pkl --train=data/groups/gld_${1}_lowvolume_vision_train.pkl --train_limit=316 --test_limit=109 --test=data/groups/gld_${1}_mediumvolume_vision_test.pkl --users $MEDIUMVOLUME_USER_IDS >> data/groups/mediumvolume.txt

python generate_negatives.py --data_file=data/groups/gld_${1}_mediumvolume_vision_train.pkl --out_file=data/groups/gld_${1}_mediumvolume_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/groups/gld_${1}_mediumvolume_vision_test.pkl --out_file=data/groups/gld_${1}_mediumvolume_vision_test_pos_neg.pkl