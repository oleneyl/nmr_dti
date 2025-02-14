"""
Adapter :: Mediating class between human - recognizable data from data_loader and
computer - recongizable numpy objects.
"""

import numpy as np
from .vocab import SentencePieceVocab, DummyVocab, SimpleSMILESVocab, DTASMILESVocab, DTAProteinVocab
from rdkit import Chem

def adapter_args(parser):
    group = parser.add_argument_group('adapter')

    group.add_argument('--protein_vocab', type=str)
    group.add_argument('--chemical_vocab', type=str)
    group.add_argument('--nmr_array_size', type=int, default=1000)
    group.add_argument('--min_ppm', type=float, default=0.0)
    group.add_argument('--max_ppm', type=float, default=10.0)
    group.add_argument('--protein_vocab_size', type=int, default=1000)
    group.add_argument('--chemical_vocab_size', type=int, default=1000)
    group.add_argument('--ignore_nmr', type=float, default=0.5)



def get_adapter(args, training=True):
    if args.protein_vocab == '__base__':
        protein_vocab = DTAProteinVocab()
    else:
        protein_vocab = SentencePieceVocab(args.protein_vocab)

    if args.chemical_vocab == '__base__':
        chemical_vocab = DTASMILESVocab()
    else:
        chemical_vocab = SentencePieceVocab(args.chemical_vocab)

    return NMRAdapter(protein_vocab, chemical_vocab, nmr_array_size=args.nmr_array_size,
                      min_ppm=args.min_ppm, max_ppm=args.max_ppm, ignore_nmr = args.ignore_nmr, training=training)


class NMRAdapter(object):
    def __init__(self, protein_vocab, chemical_vocab, nmr_array_size=1000,
                 min_ppm=0,
                 max_ppm=10, ignore_nmr=0.0, training=True):
        self.protein_vocab = protein_vocab
        self.chemical_vocab = chemical_vocab
        self.nmr_array_size = nmr_array_size
        self.min_ppm = min_ppm
        self.max_ppm = max_ppm
        self.ignore_nmr = ignore_nmr
        self.training = training

    @classmethod
    def normalize_nmr(cls, nmr_data, nmr_min, nmr_max, min_ppm, max_ppm, size, h_count=1):
        """Fit given data into min_ppm ~ max_ppm with given size.
        """

        nmr_data = np.array(nmr_data)
        nmr_data = nmr_data - np.min(nmr_data)
        nmr_data = nmr_data / np.sum(nmr_data) * h_count / (size / len(nmr_data))
        target_point = np.linspace(nmr_min, nmr_max, size)
        current_point = np.linspace(min_ppm, max_ppm, len(nmr_data))
        nmr_interp = np.interp(target_point, current_point, nmr_data, 0.0, 0.0)
        nmr_data = nmr_interp
        nmr_seq = nmr_data  # No log scale
        #nmr_seq = np.log(nmr_data)  # Log scale
        nmr_seq = (nmr_seq - np.min(nmr_seq)) * (nmr_interp > 0) + np.zeros([size])
        return nmr_seq

    def adapt(self, datum):
        protein_indices = self.protein_vocab(datum.get('sequence', [0]))
        chemical_indices = self.chemical_vocab(datum.get('smiles', [0]))
        mol = Chem.MolFromSmiles(datum.get('smiles', [0]))
        mol = Chem.AddHs(mol)
        h_count = len([i for i in mol.GetAtoms() if i.GetAtomicNum() == 1])
        if 'nmr_freq' in datum:
            if self.training and np.random.random() < self.ignore_nmr:
                nmr_values = np.zeros([self.nmr_array_size])
            else:
                nmr_values = self.normalize_nmr(datum['nmr_freq'], datum['nmr_rg'][0], datum['nmr_rg'][1],
                                        self.min_ppm, self.max_ppm, self.nmr_array_size, h_count=h_count)
        else:
            nmr_values = np.zeros([self.nmr_array_size])

        return float(datum['bind']), protein_indices, chemical_indices, nmr_values

    def __call__(self, data_loader):
        for item in data_loader:
            yield self.adapt(item)
