import torch
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel
import torch
import pickle
import soundfile as sf
from speech_reps.featurize import DeCoARFeaturizer
def setup_device(gpu_num=0):
    """Setup device."""
    device_name = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'  # Is there a GPU?
    device = torch.device(device_name)
    return device
def vq_wav2vec_featurize(wav_file, gpu_num=0, bert_model='./models/bert_kmeans.pt', vq_wav2vec_model='models/vq-wav2vec_kmeans.pt', max_size=1300000):
    device = setup_device(gpu_num=gpu_num)
    cp = torch.load(vq_wav2vec_model)
    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])
    model.eval()
    model.to(device)

    bert_path = "/".join(bert_model.split("/")[:-1])
    bert_checkpoint = bert_model.split("/")[-1]
    roberta = RobertaModel.from_pretrained(bert_path, checkpoint_file=bert_checkpoint, data_name_or_path="./")
    roberta.eval()
    roberta.to(device)

    wav, sr = sf.read(wav_file)
    assert sr == 16000

    wav = torch.from_numpy(wav).float()

    x = wav.unsqueeze(0).float().to(device)

    div = 1
    while x.size(-1) // div > max_size:
        div += 1

    xs = x.chunk(div, dim=-1)
    quantize_location = getattr(cp["args"], "vq", "encoder")
    result = []
    for x in xs:
        torch.cuda.empty_cache()
        x = model.feature_extractor(x)
        if quantize_location == "encoder":
            with torch.no_grad():
                _, idx = model.vector_quantizer.forward_idx(x)
                idx = idx.squeeze(0).cpu()
        else:
            with torch.no_grad():
                z = model.feature_aggregator(x)
                _, idx = model.vector_quantizer.forward_idx(z)
                idx = idx.squeeze(0).cpu()
        result.append(idx)

    idx = torch.cat(result, dim=0)
    seq = " ".join("-".join(map(str, a.tolist())) for a in idx)
    print(seq)

    dict_ = roberta.task.source_dictionary

    with torch.no_grad():
        words = seq.split()
        vec = [0] + [dict_.index(w) for w in words][:2046] + [2]
        x = torch.LongTensor(vec).unsqueeze(0).to(device)
        z = roberta.extract_features(x, return_all_hiddens=True)[-4:]
        feature = torch.mean(torch.stack([torch.cat([i[:,j] for i in z]).flatten() for j in range(0, z[0].shape[1])]), 0).detach().cpu()

    return feature

def decoar_featurize(wav_file, gpu_num=0, decoar_model='models/decoar-encoder-29b8e2ac.params'):
    # Load the model on GPU 0
    featurizer = DeCoARFeaturizer(decoar_model, gpu=gpu_num)
    # Returns a (time, feature) NumPy array
    data = featurizer.file_to_feats(wav_file)
    feature = torch.mean(torch.FloatTensor(data), dim=0).detach().cpu()
    return feature
# #feature = vq_wav2vec_featurize(wav_file="speech/train/fork_2_2_4.wav")
# feature = decoar_featurize(wav_file="data/gold/speech/train/fork_2_2_4.wav")
# print(feature)
# print(feature.shape)