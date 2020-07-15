import flair
import argparse
from sentence_transformers import SentenceTransformer
import pickle

def main(args):
    with open(args.language_data, "rb") as f:
        language_data = pickle.load(f, encoding='bytes')
    with open(args.vision_data, "rb") as f:
        vision_data = pickle.load(f, encoding='bytes')
    keys_l = language_data.keys()
    keys_v = vision_data.keys()

    instances = list(set(keys_l) & set(keys_v))
    objects = ["_".join(i.split("_")[:-2]) for i in instances]

    vision_data = [vision_data[i] for i in instances]
    language_data = [language_data[i] for i in instances]

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
            instance_data = []
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
        descriptions = [i[0] for i in languages]
        descriptions_sbert = sbert_embedding(descriptions)

        for i in range(len(descriptions_sbert)):
            if languages[i][1] in language_data:
                language_data[languages[i][1]].append(descriptions_sbert[i])
            else:
                language_data[languages[i][1]] = [descriptions_sbert[i]]

    elif args.method == "bert":
        languages = []
        for instance_name, language_descriptions in text_descriptions.items():
            instance_data = []
            for desc in language_descriptions:
                languages.append([desc.strip(), instance_name])
                instance_data.append(bert_embedding(desc.strip()))
            language_data.update({instance_name: instance_data})

    pickle.dump(language_data, open(args.output, "wb"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--language_data",
                        type=str,
                        required=True,
                        help="Specify path to language data!")

    parser.add_argument("--vision_data",
                        type=str,
                        required=True,
                        help="Specify path to vision data!")

    parser.add_argument("--output",
                        default="dataset.pkl",
                        type=str,
                        required=False,
                        help="Specify a output filename!")
    args = parser.parse_args()
    main(args)
