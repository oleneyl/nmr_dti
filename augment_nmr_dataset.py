from preprocess.data_reader import DataFrameReader
from preprocess.nmr_prediction_dataset import NMRPredictionDatasetReaer, retrieve_smiles
from preprocess.data_utils.atom_embedding import create_additive_embedding
from rdkit import Chem

import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm

def create_base_dataset(input_file_name, output_file_name):
    nmr_dataset_reader = NMRPredictionDatasetReaer(input_file_name)
    columns = [
        "smiles",
        "nmr_value_list",
        "mask"
    ]

    df = pd.DataFrame(columns=columns)
    for idx in range(len(nmr_dataset_reader)):
        smiles, nmr_value_list, mask = nmr_dataset_reader._manual_getitem(idx) # Protected method
        df = df.append({
            'smiles' : smiles,
            'nmr_value_list' : nmr_value_list,
            'mask' : mask
        })

    with open(output_file_name, 'wb') as f:
        pickle.dump(df, f)


def create_augmented_dataset(input_file_name, output_file_name):
    nmr_dataset_reader = NMRPredictionDatasetReaer(input_file_name, data_type='train')
    columns = [
        "smiles",
        "nmr_value_list",
        "mask"
    ]

    df = pd.DataFrame(columns=columns)

    data = []

    print(f"Initial data size : {len(nmr_dataset_reader)}")

    for idx in tqdm(range(len(nmr_dataset_reader))):
        mol = nmr_dataset_reader._cache.iloc[idx]['rdmol']
        mol = Chem.rdmolops.RemoveAllHs(mol)
        atoms = mol.GetNumAtoms()
        for atom_idx in range(atoms - 1):
            try:
                datum = nmr_dataset_reader._manual_getitem(idx, rooted_atom=atom_idx) # Protected method
            except:
                continue
            if datum is not None:
                data.append(datum)

    print(f"Total data size : {len(data)}")
    np.random.shuffle(data)

    with open(output_file_name, 'wb') as f:
        pickle.dump(data, f)


def create_augmented_atomic_dataset(input_file_name, output_file_name):
    nmr_dataset_reader = NMRPredictionDatasetReaer(input_file_name, data_type='train')
    columns = [
        "smiles",
        "nmr_value_list",
        "mask",
        "atomic"
    ]

    data = []

    print(f"Initial data size : {len(nmr_dataset_reader)}")

    for idx in tqdm(range(len(nmr_dataset_reader))):
        mol = nmr_dataset_reader._cache.iloc[idx]['rdmol']
        mol = Chem.rdmolops.RemoveAllHs(mol)
        atoms = mol.GetNumAtoms()
        for atom_idx in range(atoms - 1):
            try:
                datum = nmr_dataset_reader._manual_getitem(idx, rooted_atom=atom_idx) # Protected method
                if datum is None:
                    continue
                simplified_smiles = datum[0]
                smiles = retrieve_smiles(simplified_smiles)
                atom_embed = create_additive_embedding(smiles, simplified_smiles)
                s1, s2, s3 = datum
                datum = [s1, s2, s3, atom_embed]
            except Exception as e:
                if type(e) is not type(KeyboardInterrupt()):
                    continue
                else:
                    raise
            data.append(datum)

    print(f"Total data size : {len(data)}")
    np.random.shuffle(data)

    with open(output_file_name, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    # create_augmented_dataset('/home/meson324/dataset/nmr_mpnn/data_13C.pickle',
    #                          '/home/meson324/dataset/nmr_mpnn/data_13C_aug.pickle')
    create_augmented_atomic_dataset('/home/meson324/dataset/nmr_mpnn/data_13C.pickle',
                             '/home/meson324/dataset/nmr_mpnn/data_13C_aug_atom.pickle')
