import flair
import argparse
from sentence_transformers import SentenceTransformer
import pickle
import torch

def fix_tensor(tens):
    tens.requires_grad = False
    return tens

def main(args):
    with open(args.input, "rb") as f:
        text_descriptions = pickle.load(f, encoding='bytes')
    language_data =dict()
    document_embeddings = flair.embeddings.DocumentPoolEmbeddings([flair.embeddings.BertEmbeddings()])
    def bert_embedding(sentence):
        sentence = flair.data.Sentence(sentence, use_tokenizer=True)
        document_embeddings.embed(sentence)
        return sentence.get_embedding().cpu()

    def sbert_embedding(sentences):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        return model.encode(sentences)

    instance_names = []
    languages = []
    descriptions = []
    object_names = []
    descriptions_features = []

    if args.method == "sbert":
        for instance_name, language_descriptions in text_descriptions.items():
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
                instance_names.append(instance_name)
                object_name = "_".join(instance_name.split("_")[:-2])
                object_names.append(object_name)
        descriptions = [i[0] for i in languages]
        descriptions_features = sbert_embedding(descriptions)

        for i in range(len(descriptions_features)):
            descriptions_features[i] = torch.tensor(descriptions_features[i])

    elif args.method == "bert":
        languages = []
        for instance_name, language_descriptions in text_descriptions.items():
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
                instance_names.append(instance_name)
                object_name = "_".join(instance_name.split("_")[:-2])
                object_names.append(object_name)
                descriptions_features.append(fix_tensor(bert_embedding(desc.strip())))

    language_dict = {"instance_names": instance_names, "object_names": object_names,
                     "embedded_vectors": descriptions_features, "descriptions": descriptions}

    pickle.dump(language_dict, open(args.output, "wb"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--input",
                        type=str,
                        required=True,
                        help="Specify a input filename!")

    parser.add_argument("--method",
                        default="sbert",
                        type=str,
                        required=False,
                        help="Specify an embedding method!")

    parser.add_argument("--output",
                        default="text_embeddings.pkl",
                        type=str,
                        required=False,
                        help="Specify a output filename!")
    args = parser.parse_args()
    main(args)
