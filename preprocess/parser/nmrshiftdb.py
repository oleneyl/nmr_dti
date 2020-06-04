import pickle
import pandas as pd
import rdkit
import os

NMRSHIFTDB_DATASET_PATH = "../datasets"


def show_dataset():
    with open(os.path.join(NMRSHIFTDB_DATASET_PATH, "data_1H.pickle"), 'rb') as f:
        data = pickle.load(f)

    return data

data = show_dataset()
print(type(data['train_df']))