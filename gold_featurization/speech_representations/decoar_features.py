from speech_reps.featurize import DeCoARFeaturizer
import os
import pickle
from tqdm import tqdm
import torch

# Load the model on GPU 0
featurizer = DeCoARFeaturizer('artifacts/checkpoint_decoar.pt', gpu=0)
# Returns a (time, feature) NumPy array
speech_dict = dict()
data_path = "./speech/"


for file_name in tqdm(os.listdir(data_path)):
  wav_path = os.path.join(data_path, file_name)
  seq_features = featurizer.file_to_feats(wav_path)
  features = torch.mean(torch.tensor(seq_features), dim=0)
  print(features.shape)
  speech_dict[file_name.replace(".wav", "")] = features

with open('decoar_features.pkl', 'wb') as f:
    pickle.dump(speech_dict, f)
