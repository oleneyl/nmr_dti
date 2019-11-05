from collections import OrderedDict
import pickle
import json
import os
import numpy as np
import math

KIBA_DATASET_PATH = '/DATA/meson324/DeepDTA/kiba'
DAVIS_DATASET_PATH = '/DATA/meson324/DeepDTA/davis'

def get_kiba_dataset(dir_path=KIBA_DATASET_PATH, as_binary=False):
    train, test = data_from_deep_dta(dir_path, as_binary=as_binary)
    # Train - valid split
    valid = train[0]
    train = sum(train[1:], [])

    return train, valid, test


def get_davis_dataset(dir_path=DAVIS_DATASET_PATH, as_binary=False):
    def to_value(bind):
        return 9 - math.log10(bind)
    train, test = data_from_deep_dta(dir_path, as_binary=as_binary, bind_callback=to_value)
    # Train - valid split
    valid = train[0]
    train = sum(train[1:], [])

    return train, valid, test


def data_from_deep_dta(dir_path, as_binary=False, bind_callback=None):
    PROTEIN_FILE_NAME = 'proteins.txt'
    LIGAND_FILE_NAME = 'ligands_can.txt'
    BINARY_CRITERION = 12.1  # https://arxiv.org/pdf/1908.06760.pdf page 10

    with open(os.path.join(dir_path, PROTEIN_FILE_NAME)) as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
        proteins_key = [x for x in proteins]

    with open(os.path.join(dir_path, LIGAND_FILE_NAME)) as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
        ligands_key = [x for x in ligands]

    with open(os.path.join(dir_path, 'Y'), 'rb') as f:
        matrix = pickle.load(f, encoding='latin1')

    label_row_indices, label_col_indices = np.where(np.isnan(matrix)==False)

    XD = []
    XT = []
    for d in ligands.keys():
        XD.append(ligands[d])

    for t in proteins.keys():
        XT.append(proteins[t])
    
    # Shuffle
    test_fold = json.load(open(os.path.join(dir_path, "folds/test_fold_setting1.txt")))
    train_folds = json.load(open(os.path.join(dir_path, "folds/train_fold_setting1.txt")))

    def create_dataset_from_fold(fold):
        from tqdm import tqdm
        dataset = []
        for idx in fold:
            r = label_row_indices[idx]
            c = label_col_indices[idx]
            ligand_id = ligands_key[int(r)]
            protein_id = proteins_key[int(c)]
            ligand_seq = XD[r]
            protein_seq = XT[c]
            bind = matrix[r][c]

            if bind_callback:
                bind = bind_callback(bind)

            if as_binary:
                bind = True if (bind > BINARY_CRITERION) else False

            dataset.append({
                'chembl_id': ligand_id,
                'uniprot_id': protein_id,
                'sequence': protein_seq,
                'smiles': ligand_seq,
                'bind': bind
            })

        return dataset

    train_dataset = [create_dataset_from_fold(fold) for fold in train_folds]
    test_dataset = create_dataset_from_fold(test_fold)
    return train_dataset, test_dataset

