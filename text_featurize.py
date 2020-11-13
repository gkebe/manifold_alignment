import torch
import flair

def fix_tensor(tens):
    tens.requires_grad = False
    return tens
def bert_embedding(sentence):
    document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])
    sentence = flair.data.Sentence(sentence.strip(), use_tokenizer=True)
    document_embeddings.embed(sentence)
    return fix_tensor(sentence.get_embedding())

# #feature = vq_wav2vec_featurize(wav_file="speech/train/fork_2_2_4.wav")
# feature = bert_embedding("it's a pair of glasses on the table")
# print(feature)
# print(feature.shape)