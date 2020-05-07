"""
Purpose : parse ibm data and create dump of usable dataset!
Remember only - need - values are protein, chemical, (nmr) values.
"""
import os

from preprocess.parser.nmr_base import get_nmr_engine
import pubchempy as pcp

IBM_DATASET_PATH = '/DATA/meson324/InterpretableDTIP/data'


def get_ibm_data_reader():
    return (IBMDataReader(os.path.join(IBM_DATASET_PATH, 'train')),
            IBMDataReader(os.path.join(IBM_DATASET_PATH, 'dev')),
            IBMDataReader(os.path.join(IBM_DATASET_PATH, 'test')))


class IBMDataReader(object):
    def __init__(self, path):
        self.base_path = path
        self.chemical_map = {}
        self.protein_map = {}

    def load_base_info(self):
        chemical_map = {}
        with open(os.path.join(self.base_path, 'chem')) as f_chem:
            with open(os.path.join(self.base_path, 'chem.repr')) as f_chem_smile:
                for chem, smile in zip(f_chem, f_chem_smile):
                    chemical_map[chem.strip('\n')] = smile.strip('\n')

        protein_map = {}
        with open(os.path.join(self.base_path, 'protein.vocab')) as f:
            protein_vocab = f.read().split('\n')
        with open(os.path.join(self.base_path, 'protein')) as f_protein:
            with open(os.path.join(self.base_path, 'protein.repr')) as f_protein_repr:
                for protein, protein_repr in zip(f_protein, f_protein_repr):
                    protein_map[protein.strip('\n')] = ''.join([protein_vocab[int(idx)]
                                                                for idx in protein_repr.strip('\n').split(' ')])

        self.chemical_map = chemical_map
        self.protein_map = protein_map

    def create_dataset(self, mix_nmr=False):
        def get_dataset_in_file(fname, binding):
            if mix_nmr:
                nmr_engine = get_nmr_engine()
            dataset = []
            with open(os.path.join(self.base_path, fname)) as f_pos:
                for line in f_pos:
                    _, chemical_idx, __, protein_idx = line.strip('\n').split(',')
                    datum = {
                        'uniprot_id': protein_idx,
                        'pubchem_id': chemical_idx,
                        'sequence': self.protein_map[protein_idx],
                        'smiles': self.chemical_map[chemical_idx],
                        'bind': binding
                    }
                    if mix_nmr:
                        datum['nmr'] = nmr_engine()
                        pcp.Compound.from_cid(chemical_idx)
                    dataset.append(datum)
            return dataset

        self.load_base_info()
        positives = get_dataset_in_file('edges.pos', True)
        negatives = get_dataset_in_file('edges.neg', False)

        return positives, negatives



