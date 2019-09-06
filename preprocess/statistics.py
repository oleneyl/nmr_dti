"""
statistics.py

For inspect datasets given by various format, and help choosing hyperparameters.
"""
from .data_reader import JSONDataReader
from collections import Counter

import numpy as np
from tqdm import tqdm
from .data_utils.numpy_adapter import SentencePieceVocab
from .kernel import get_conf


def show_protein_statistics(data_file_name, protein_vocab):
    proteins = JSONDataReader(data_file_name)   # Positive sample only is recommended

    #Tokenize
    tokenizer = SentencePieceVocab(protein_vocab)

    # Length
    length_counter = Counter()
    pt_set = set([])
    for pt in tqdm(proteins):
        pt_set.update([pt['sequence']])

    for pt in tqdm(pt_set):
        length_counter.update([len(tokenizer(pt))])

    commons = length_counter.most_common()
    commons.sort(key=lambda x: x[0])

    length_dist, count_dist = list(zip(*commons))
    return length_dist, count_dist


def show_distribution_statistics(data_file_name):
    dataset = JSONDataReader(data_file_name)

    labels = [0,0]
    for datum in dataset:
        labels[1 if datum['bind'] else 0] += 1

    max_acc = max(labels[0]/sum(labels), labels[1]/sum(labels))

    print(f'Positive {labels[0]}, Negative {labels[1]}, reachable random-selection accuracy {max_acc}')


def show_overlap(train_file, valid_file):
    train = JSONDataReader(train_file)
    valid = JSONDataReader(valid_file)

    def get_set(dataset):
        chemical, protein = Counter(), Counter()
        for datum in dataset:
            chemical.update([datum['smiles']])
            protein.update(([datum['sequence']]))
        return chemical, protein

    train_result, valid_result = get_set(train), get_set(valid)

    print(f'Chemicals : Train {len(train_result[0])} | Valid {len(valid_result[0])}')
    print(f'Proteins : Train {len(train_result[1])} | Valid {len(valid_result[1])}')

    print(f'Chemical overlap : {len(set(train_result[0].keys()) & set(valid_result[0].keys()))}')
    print(f'Protein overlap : {len(set(train_result[1].keys()) & set(valid_result[1].keys()))}')

    return train_result, valid_result