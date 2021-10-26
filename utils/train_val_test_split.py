import pickle
from datasets import GLData
import pandas as pd

for split in ["train", "dev", "test"]:
    a = pickle.load(open(f"data/gld_files_vision_{split}.pkl", "rb"))
    files = [i[0] for i in a]
    files_df = pd.DataFrame(files, columns=["audio_files"])
    files_df.to_csv(f"data/gld_{split}.csv", index=False)
