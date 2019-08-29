"""
Adapter :: Mediating class between human - recognizable data from data_loader and
computer - recongizable numpy objects.
"""

import numpy as np
import sentencepiece as spm

class Vocab(object):
    """Vocab converts given sequence into list - of - index
    """
    def __init__(self):
        pass

    def __call__(self, sequence):
        return self.encode(sequence)

    def encode(self, sequence):
        raise NotImplementedError

    def decode(self, indices_array):
        raise NotImplementedError


class SentencePieceVocab(Vocab):
    def __init__(self, vocab_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab_file)

    def encode(self, sequence):
        return self.sp.encode_as_ids(sequence)

    def decode(self, indices_array):
        return self.sp.decode_ids(indices_array)


class NMRAdapter():
    def __init__(self, protein_vocab, chemical_vocab, nmr_array_size=1000,
                 protein_sequence_size=1000,
                 chemical_sequence_size=100,
                 min_ppm=0,
                 max_ppm=10):
        self.protein_vocab = protein_vocab
        self.chemical_vocab = chemical_vocab
        self.nmr_array_size = nmr_array_size
        self.min_ppm = min_ppm
        self.max_ppm = max_ppm

    @classmethod
    def normalize_nmr(cls, nmr_data, nmr_min, nmr_max, min_ppm, max_ppm, size):
        """Fit given data into min_ppm ~ max_ppm with given size.
        """
        target_point = np.arange(nmr_min, nmr_max, (nmr_max - nmr_min)/size)
        current_point = np.arange(min_ppm, max_ppm, (max_ppm-min_ppm)/size)
        return np.interp(target_point, current_point, nmr_data, 0.0, 0.0)

    def adapt(self, datum):
        protein_indices = self.protein_vocab(datum['sequence'])
        chemical_indices = self.chemical_vocab(datum['smiles'])
        nmr_values = self.normalize_nmr(datum['nmr_freq'], datum['nmr_rg'][0], datum['nmr_rg'][1],
                                        self.min_ppm, self.max_ppm, self.nmr_array_size)

        return datum['binds'], protein_indices, chemical_indices, nmr_values

    def __call__(self, data_loader):
        for item in data_loader:
            yield self.adapt(item)
