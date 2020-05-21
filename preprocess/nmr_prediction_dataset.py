import numpy as np
import pandas as pd
import pickle
import rdkit
from rdkit import Chem
import string
import tensorflow as tf

SMILES_SIMPLIFY_MAP = {
    "Cl": "L",
    "Br": "R"
}


def simplify_smiles(smiles):
    for k, v in SMILES_SIMPLIFY_MAP.items():
        smiles = smiles.replace(k, v)

    return smiles


def retrieve_smiles(smiles):
    for k, v in SMILES_SIMPLIFY_MAP.items():
        smiles = smiles.replace(v, k)

    return smiles


class NMRPredictionDatasetReaer():
    def __init__(self, fname, data_type):
        self.fname = fname
        self._cache = []
        self.cache_data(data_type)

    def cache_data(self, data_type):
        with open(self.fname, 'rb') as f:
            dataset = pickle.load(f)

        if data_type == 'train':
            self._cache = dataset['train_df']
        elif data_type == 'test' or data_type == 'valid':
            self._cache = dataset['test_df']

    def __getitem__(self, idx):
        item = self._cache.iloc[idx]
        return NMRPredictionDatasetReaer._parse_item_into_input(item, target_atom='C')

    def _manual_getitem(self, idx, log=False, rooted_atom=-1):
        item = self._cache.iloc[idx]
        return NMRPredictionDatasetReaer._parse_item_into_input(item, target_atom='C', log=log, rooted_atom=rooted_atom)

    def _validate(self, smiles, value, target_atom):
        # count carbon
        atom_count = 0
        for ch in smiles:
            if ch.upper() == target_atom:
                atom_count += 1

        return (atom_count <= len(value))

    @classmethod
    def _parse_item_into_input(cls, item, target_atom='C', log=False, rooted_atom=-1):
        def standardize_value(nmr_pick):
            MEAN = 97.53
            STD = 51.53
            return (nmr_pick - MEAN)

        mol = item['rdmol']
        nmr_value = item['value'][0]
        mol_origin = mol

        # Transform given molecule into SMILES without Hydrogen
        mol = Chem.rdmolops.RemoveAllHs(mol)
        smiles = Chem.MolToSmiles(mol, allHsExplicit=False, rootedAtAtom=rooted_atom)
        mol = Chem.MolFromSmiles(smiles)

        align_list = mol_origin.GetSubstructMatch(mol)

        # If alignment fail
        if len(align_list) == 0:
            return None

        # Create mask
        nmr_value_list = [0.0 for i in range(len(smiles))]

        n_atom = mol.GetNumAtoms()
        if log:
            print(smiles)
            print(simplify_smiles(smiles))
            print(align_list)
        smiles = simplify_smiles(smiles)
        atom_count = 0

        # Input validation
        if not cls._validate(cls, smiles, nmr_value, target_atom):
            return None

        for idx, ch in enumerate(smiles):
            if ch.upper() == target_atom:
                try:
                    nmr_value_list[idx] = standardize_value(nmr_value[align_list[atom_count]])
                except IndexError:
                    # This error may raise when atomic index parsing failed.
                    raise

            if ch in (string.ascii_uppercase + string.ascii_lowercase) and ch is not 'H':
                atom_count += 1

        mask = [(0 if val == 0.0 else 1) for val in nmr_value_list]

        return smiles, nmr_value_list, mask

    def __iter__(self):
        if len(self._cache) > 0:
            for idx in range(len(self._cache)):
                yield self[idx]
        else:
            raise IndexError("NMRPredictionDatasetReader must be cached")

    def __len__(self):
        return len(self._cache)