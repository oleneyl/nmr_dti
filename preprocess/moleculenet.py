import numpy as np
import deepchem
from .nmr_prediction_dataset import get_positional_information, create_mask
from .data_utils.vocab import AtomOrbitalVocab
from .extensions.loader import load_qm9

class MoleculeNetDatasetLoader(object):
    def __init__(self, data_type, batch_size=1,
                 chemical_sequence_length=256,
                 training=True):

        self.batch_size = batch_size
        self.data_type = data_type

        self.chemical_sequence_length = chemical_sequence_length
        self.training = training
        self.vocab = AtomOrbitalVocab()
        self.batch_size = batch_size

        task, dataset, tf = self.load_dataset()
        self._task = task
        self._transformer = tf

        # Dataset type check
        if data_type == 'train':
            self._dataset = dataset[0]
        elif data_type == 'valid':
            self._dataset = dataset[1]
        elif data_type == 'test':
            self._dataset = dataset[2]
        else:
            raise TypeError('Dataset type must be train, valid or test.')

    def load_dataset(self):
        # Load dataset by using deepchem package
        raise NotImplementedError("Must implement which dataset to be loaded.")

    def unpack_item(self, listed_item):
        # unpack given item format into (feature, label, smiles) tuple
        raise NotImplementedError("Must implement how to unpack item.")

    def reset(self):
        pass

    def get_molecule(self, listed_item):
        # If molecule was already given, return molecule. If not, return None.
        return None

    def get_conformer(self, listed_item):
        # If conformer was already given, return conformer. If not, return None.
        return None

    def __iter__(self):
        batch = []
        for item in self._dataset.itersamples():
            listed_item = list(item)
            feature, label, smiles = self.unpack_item(listed_item)

            _intermediate = get_positional_information(smiles,
                                                       conformer=self.get_conformer(listed_item),
                                                       mol=self.get_molecule(listed_item))
            if _intermediate is None:
                continue
            else:
                position_matrix, direction_matrix, embedding_list, atom_to_orbital = _intermediate
            pad_mask = [0.0 for x in range(len(position_matrix))]
            pad_mask = pad_mask + [1.0 for x in range(self.chemical_sequence_length)]

            position_matrix = position_matrix + [[0, 0, 0] for x in range(self.chemical_sequence_length)]
            direction_matrix = direction_matrix + [[[0, 0, 0], [0, 0, 0], [0, 0, 0]] for x in
                                                   range(self.chemical_sequence_length)]
            position_matrix = np.array(position_matrix[:self.chemical_sequence_length])
            direction_matrix = np.array(direction_matrix[:self.chemical_sequence_length])

            distance, angular_distance = create_mask(position_matrix, direction_matrix)

            atomic_length = self.chemical_sequence_length // 2

            embedding_list = self.vocab(embedding_list) + [self.vocab.ENDL for x in
                                                           range(self.chemical_sequence_length)]
            orbital_matrix = np.zeros([atomic_length, self.chemical_sequence_length])

            for a_idx, o_idx in atom_to_orbital:
                if o_idx >= self.chemical_sequence_length or a_idx >= atomic_length:
                    continue
                else:
                    orbital_matrix[a_idx, o_idx] = 1

            packet = [embedding_list[:self.chemical_sequence_length],
                      distance,
                      angular_distance,
                      orbital_matrix,
                      label,
                      pad_mask[:self.chemical_sequence_length]]

            batch.append(packet)

            if len(batch) == self.batch_size:
                yield [np.array(x) for x in zip(*batch)]
                batch = []


class Tox21DatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_tox21()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class BBBPDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_bbbp()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class ToxCastDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_toxcast()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class SIDERDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_sider()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class ClinToxDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_clintox()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class MUVDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_muv()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class HIVDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_hiv()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class BACEClassificationDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_bace_classification()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class QM8DatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return deepchem.molnet.load_qm8()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles


class QM9DatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return load_qm9()

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        return feature, label, smiles

    def get_conformer(self, listed_item):
        # If conformer was already given, return conformer. If not, return None.
        feature, label, _, smiles = listed_item
        '''
        print(smiles)
        print(feature[0])
        print(feature[0].GetConformers())
        print(feature[0].GetConformers()[0].GetPositions().shape)
        '''
        return feature[0].GetConformers()[0]

    def get_molecule(self, listed_item):
        # If conformer was already given, return conformer. If not, return None.
        feature, label, _, smiles = listed_item
        return feature[0]


class QM9SingleDatasetLoader(MoleculeNetDatasetLoader):
    def load_dataset(self):
        return load_qm9(move_mean=False)

    def unpack_item(self, listed_item):
        # _ : existence
        feature, label, _, smiles = listed_item
        # HOMO
        return feature, label[2] * 27.2114, smiles

    def get_conformer(self, listed_item):
        # If conformer was already given, return conformer. If not, return None.
        feature, label, _, smiles = listed_item
        '''
        print(smiles)
        print(feature[0])
        print(feature[0].GetConformers())
        print(feature[0].GetConformers()[0].GetPositions().shape)
        '''
        return feature[0].GetConformers()[0]

    def get_molecule(self, listed_item):
        # If conformer was already given, return conformer. If not, return None.
        feature, label, _, smiles = listed_item
        return feature[0]



AVAILABLE_MOLECULENET_TASK = {
    'Tox21': Tox21DatasetLoader,
    'BBBP': BBBPDatasetLoader,
    'ToxCast': ToxCastDatasetLoader,
    'SIDER': SIDERDatasetLoader,
    'ClinTox': ClinToxDatasetLoader,
    'MUV': MUVDatasetLoader,
    'HIV': HIVDatasetLoader,
    'BACE': BACEClassificationDatasetLoader,
    'QM8': QM8DatasetLoader,
    'QM9': QM9DatasetLoader,
    'QM9Single': QM9SingleDatasetLoader,
}


def get_dataset_loader(task, data_type, batch_size=1,
                 chemical_sequence_length=256,
                 training=True):
    if task not in AVAILABLE_MOLECULENET_TASK:
        raise TypeError(f'Given task f{task} does not exist.')

    return AVAILABLE_MOLECULENET_TASK[task](data_type,
                                            batch_size=batch_size,
                                            chemical_sequence_length=chemical_sequence_length,
                                            training=training)