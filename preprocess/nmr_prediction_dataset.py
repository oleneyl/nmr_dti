import numpy as np
import pandas as pd
import pickle
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import string

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

            if ch in (string.ascii_uppercase + string.ascii_lowercase) and ch != 'H':
                atom_count += 1

        mask = [(0 if val == 0.0 else 1) for val in nmr_value_list]

        # Extract only atom


        return smiles, nmr_value_list, mask

    def __iter__(self):
        if len(self._cache) > 0:
            for idx in range(len(self._cache)):
                yield self[idx]
        else:
            raise IndexError("NMRPredictionDatasetReader must be cached")

    def __len__(self):
        return len(self._cache)


class NMRPredictionAtomicDatasetReader(NMRPredictionDatasetReaer):
    def __getitem__(self, idx):
        item = self._cache.iloc[idx]
        return NMRPredictionAtomicDatasetReader._parse_item_into_input(item, target_atom='C')

    def _manual_getitem(self, idx, log=False, rooted_atom=-1):
        item = self._cache.iloc[idx]
        return NMRPredictionAtomicDatasetReader._parse_item_into_input(item, target_atom='C', log=log, rooted_atom=rooted_atom)

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
        align_list = mol_origin.GetSubstructMatch(mol)
        smiles = Chem.MolToSmiles(mol, allHsExplicit=False, rootedAtAtom=rooted_atom)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        conformer = mol.GetConformers()
        if len(conformer) == 0:
            return None
        else:
            conformer = conformer[0]

        # If alignment fail
        if len(align_list) == 0:
            return None

        # Input validation
        if not cls._validate(cls, smiles, nmr_value, target_atom):
            return None

        # Conformation
        position_raw = conformer.GetPositions()


        # NMR value
        nmr_value_list = []
        atom_to_orbital = []

        direction_matrix = []
        position_matrix = []
        embedding_list = []

        orbital_index = 0
        for idx, atom in enumerate(mol.GetAtoms()):
            # Orbital Map
            if atom.GetAtomicNum() == 1:
                atom_to_orbital.append([idx, orbital_index])
                direction_matrix.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

                position_matrix.append(position_raw[idx])
                embedding_list.append(atom.GetSymbol() + 's')

                orbital_index += 1
            else:
                for i in range(4):
                    atom_to_orbital.append([idx, orbital_index + i])
                    position_matrix.append(position_raw[idx])

                direction_matrix.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                direction_matrix.append([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
                direction_matrix.append([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
                direction_matrix.append([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

                for orb in ['s', 'px', 'py', 'pz']:
                    embedding_list.append(atom.GetSymbol() + orb)

                orbital_index += 4

            # NMR value mapping
            if atom.GetAtomicNum() == 6:
                nmr_value_list.append(nmr_value[align_list[idx]])
            else:
                nmr_value_list.append(0)

        # Mask
        mask = [(0 if val == 0.0 else 1) for val in nmr_value_list]

        return position_matrix, direction_matrix, embedding_list, atom_to_orbital, nmr_value_list, mask


def get_positional_information(smiles, conformer=None, mol=None, repeat=1):
    # If Mol not given, Transform given molecule into SMILES without Hydrogen
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)

    # If SMILES parsing fail, return None
    if mol is None:
        return None

    add_hydrogen = True

    for idx, atom in enumerate(mol.GetAtoms()):
        # Orbital Map
        if atom.GetAtomicNum() == 1:
            add_hydrogen = False

    if add_hydrogen:
        mol = Chem.AddHs(mol)

    # If conformer not given, create conformer
    if conformer is None:
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        conformer = mol.GetConformers()
        if len(conformer) == 0:
            return None
        else:
            conformer = conformer[0]

    # Conformation
    position_raw = conformer.GetPositions()
    '''
    print('--')
    print(smiles)
    print(len(mol.GetAtoms()))

    print(position_raw.shape)
    print(len(mol.GetAtoms()))
    print('--')
    '''

    # NMR value
    atom_to_orbital = []

    direction_matrix = []
    position_matrix = []
    embedding_list = []

    if mol.GetNumAtoms() != position_raw.shape[0]:
        print('--')
        print(smiles)
        print(len(mol.GetAtoms()))

        print(position_raw.shape)
        print('--')
        return None
        # raise ValueError('Size not matching, molecule and SMILES.')

    orbital_index = 0
    for idx, atom in enumerate(mol.GetAtoms()):
        # Orbital Map
        if atom.GetAtomicNum() == 1:
            atom_to_orbital.append([idx, orbital_index])
            direction_matrix.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

            position_matrix.append(position_raw[idx])
            embedding_list.append(atom.GetSymbol() + 's')

            orbital_index += 1
        else:
            for i in range(repeat):
                atom_to_orbital.append([idx, orbital_index + i])
                position_matrix.append(position_raw[idx])

            for i in range(repeat):
                input_direction = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if i > 0:
                    input_direction[0][i-1] = 1
                direction_matrix.append(input_direction)

            for orb in ['s', 'px', 'py', 'pz'][:repeat]:
                embedding_list.append(atom.GetSymbol() + orb)

            orbital_index += repeat

    return position_matrix, direction_matrix, embedding_list, atom_to_orbital


def create_mask(position, direction):
    # direction :
    # [ [], [], [] ] normalized vectors (max 3)
    # print(position.shape, direction.shape)
    position = np.expand_dims(position, axis=0)
    position_t = np.transpose(position, (1, 0, 2))

    vector = position - position_t
    distance = np.sqrt(np.sum(np.square(vector), axis=-1))

    distance = distance + np.eye(distance.shape[0]) * 1e9

    mask = (np.sum(direction, axis=-1) != 0).astype(np.int32)  # [L, 3]

    extended_vector = np.expand_dims(vector, axis=2)
    extended_distance = np.expand_dims(distance, axis=-1)

    # Along x-axis

    dot_x = np.sum(extended_vector * np.expand_dims(direction, axis=0), axis=-1)
    dot_y = np.sum(extended_vector * np.expand_dims(direction, axis=1), axis=-1)
    dot_x = dot_x / extended_distance * np.expand_dims(mask, axis=0) + (1 - np.expand_dims(mask, axis=0))  # [L, L, 3]
    dot_y = dot_y / extended_distance * np.expand_dims(mask, axis=1) + (1 - np.expand_dims(mask, axis=1))  # [L, L, 3]

    angular_distance = np.prod(dot_x * dot_y, axis=-1)
    angular_distance = angular_distance * (1 - np.eye(dot_x.shape[0])) + np.eye(dot_x.shape[0])

    return distance, angular_distance
