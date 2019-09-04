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
