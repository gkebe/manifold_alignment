# manifold_alignment

python split_data_pre_loader.py --data=data/gld_bert_transcriptions_vision_tensors.pkl --train=data/gld_bert_transcriptions_vision_train.pkl --test=data/gld_bert_transcriptions_vision_test.pkl --user USER_ID

python generate_negatives.py --data_file=data/gld_bert_transcriptions_vision_train.pkl --out_file=data/gld_bert_transcriptions_vision_train_pos_neg.pkl

python generate_negatives.py --data_file=data/gld_bert_transcriptions_vision_test.pkl --out_file=data/gld_bert_transcriptions_vision_test_pos_neg.pkl

python train_cosine.py --experiment_name=gld_bert_transcriptions --epochs=60 --train_data=data/gld_bert_transcriptions_vision_train.pkl --gpu_num 0 --pos_neg_examples_file=data/gld_bert_transcriptions_vision_train_pos_neg.pkl

python evaluate_cosine.py --experiment_name=gld_bert_transcriptions --test_data_path=output/gld_bert_transcriptions_vision_test.pkl --gpu_num 0 --pos_neg_examples_file=data/gld_bert_transcriptions_vision_test_pos_neg.pkl
