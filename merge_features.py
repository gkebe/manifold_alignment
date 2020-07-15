import flair
import argparse
from sentence_transformers import SentenceTransformer
import pickle

def fix_tensor(tens):
    tens.requires_grad = False
    return tens

def main(args):
    with open(args.language_data, "rb") as f:
        language_data = pickle.load(f, encoding='bytes')
    with open(args.vision_data, "rb") as f:
        vision_data = pickle.load(f, encoding='bytes')
    keys_l = language_data.keys()
    keys_v = vision_data.keys()

    instances = list(set(keys_l) & set(keys_v))
    objects = ["_".join(i.split("_")[:-2]) for i in instances]

    data = dict()
    data["instance_names"] = []
    data["object_names"] = []
    data["language_data"] = []
    data["vision_data"] = []

    for i in range(len(instances)):
        for lang in language_data[instances[i]]:
            data["instance_names"].append(instances[i])
            data["object_names"].append(objects[i])
            data["language_data"].append(fix_tensor(lang))
            data["vision_data"].append(fix_tensor(vision_data[instances[i]]))

    pickle.dump(data, open(args.output, "wb"))
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
