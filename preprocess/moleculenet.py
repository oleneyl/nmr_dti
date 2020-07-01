import numpy as np
import deepchem
from .nmr_prediction_dataset import get_positional_information, create_mask
from .data_utils.vocab import AtomOrbitalVocab


class Tox21DatasetLoader():
    def __init__(self, data_type, batch_size=1,
                 chemical_sequence_length=256,
                 training=True):

        self.batch_size = batch_size
        self.data_type = data_type

        self.chemical_sequence_length = chemical_sequence_length
        self.training = training
        self.vocab = AtomOrbitalVocab()
        task, dataset, tf = deepchem.molnet.load_tox21()
        self.batch_size = batch_size
        self._task = task

        if data_type == 'train':
            self._dataset = dataset[0]
        elif data_type == 'valid':
            self._dataset = dataset[1]
        elif data_type == 'test':
            self._dataset = dataset[2]
        else:
            raise TypeError('Dataset type must be train, valid or test')

        self._transformer = tf

    def reset(self):
        pass

    def __iter__(self):
        batch = []
        for item in self._dataset.itersamples():
            listed_item = list(item)
            feature, label, existance, smiles = listed_item

            _intermediate = get_positional_information(smiles)
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

