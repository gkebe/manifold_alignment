from fairseq.models.roberta import RobertaModel
import torch
import pickle
from tqdm import tqdm
roberta = RobertaModel.from_pretrained('./models', checkpoint_file='bert_kmeans.pt', data_name_or_path="./")
roberta.eval()
roberta.cuda()
file1 = open('results/train.src', 'r')
Lines = file1.readlines()
dict_ = roberta.task.source_dictionary
speech_dict = dict()

with torch.no_grad():
	for line in tqdm(Lines):
		if ".wav" in line:
			fname = f"speech_16_/{line.strip()}"
		else:
			words = line.split()
			vec = [0] + [dict_.index(w) for w in words][:2046] + [2]
			x = torch.LongTensor(vec).unsqueeze(0).cuda()
			z = roberta.extract_features(x, return_all_hiddens=True)[-4:]
			speech_dict[fname.split("/")[-1].replace(".wav", "")]=torch.mean(torch.stack([torch.cat([i[:,j] for i in z]).flatten() for j in range(0, z[0].shape[1])]), 0).detach().cpu()
with open('vq-wav2vec_bert_features_mean.pkl', 'wb') as f:
	pickle.dump(speech_dict, f)
