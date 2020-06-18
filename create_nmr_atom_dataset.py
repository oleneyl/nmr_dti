from preprocess.data_reader import DataFrameReader
from preprocess.nmr_prediction_dataset import NMRPredictionAtomicDatasetReader
from preprocess.data_utils.atom_embedding import create_additive_embedding
from rdkit import Chem

import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm



def create_cached_dataset(input_file_name, output_file_name):
    nmr_dataset_reader = NMRPredictionAtomicDatasetReader(input_file_name, data_type='train')
    columns = [
        "position_matrix",
        "direction_matrix",
        "embedding_list",
        "atom_to_orbital",
        "nmr_value_list",
        "mask"
    ]

    df = pd.DataFrame(columns=columns)

    data = []

    print(f"Initial data size : {len(nmr_dataset_reader)}")

    for idx in tqdm(range(len(nmr_dataset_reader))):
        item = nmr_dataset_reader[idx]
        if item is not None:
            data.append(item)

    print(f"Total data size : {len(data)}")

    with open(output_file_name, 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # create_augmented_dataset('/home/meson324/dataset/nmr_mpnn/data_13C.pickle',
    #                          '/home/meson324/dataset/nmr_mpnn/data_13C_aug.pickle')
    create_cached_dataset('/home/meson324/dataset/nmr_mpnn/data_13C.pickle',
                             '/home/meson324/dataset/nmr_mpnn/data_13C_atomic.pickle')
