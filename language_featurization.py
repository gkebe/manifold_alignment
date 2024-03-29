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
        return sentence.get_embedding()

    def sbert_embedding(sentences):
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        return model.encode(sentences)

    if args.method == "sbert":
        languages = []
        for instance_name, language_descriptions in text_descriptions.items():
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
        descriptions = [i[0] for i in languages]
        descriptions_sbert = sbert_embedding(descriptions)

        for i in range(len(descriptions_sbert)):
            if languages[i][1] in language_data:
                language_data[languages[i][1]].append(torch.tensor(descriptions_sbert[i]))
            else:
                language_data[languages[i][1]] = [torch.tensor(descriptions_sbert[i])]

    elif args.method == "bert":
        languages = []
        for instance_name, language_descriptions in text_descriptions.items():
            instance_data = []
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
                instance_data.append(fix_tensor(bert_embedding(desc.strip())))
            language_data.update({instance_name: instance_data})

    pickle.dump(language_data, open(args.output, "wb"))
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
