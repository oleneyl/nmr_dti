from ..data_reader import JSONDataReader, NMRDataReader, RestrictiveNMRDataReader
from ..kernel import get_conf
from .numpy_adapter import get_adapter
import numpy as np
import os
import json


def data_loader_args(parser):
    base_conf = get_conf()
    group = parser.add_argument_group('data_loader')
    group.add_argument('--dataset_dir', type=str, default='')
    group.add_argument('--train_file_name', type=str, default='')
    group.add_argument('--valid_file_name', type=str, default='')
    group.add_argument('--test_file_name', type=str, default='')
    group.add_argument('--nmr_dir', type=str)
    group.add_argument('--batch_size', type=int)
    group.add_argument('--protein_sequence_length', type=int, default=256)
    group.add_argument('--chemical_sequence_length', type=int, default=256)
    group.add_argument('--as_score', action='store_true')


def get_data_loader(args):
    train = args.train_file_name
    valid = args.valid_file_name
    test = args.test_file_name

    if len(args.dataset_dir) > 0:
        with open(os.path.join(args.dataset_dir, 'config.json')) as f:
            config = json.load(f)
        train = os.path.join(args.dataset_dir, config['train_set'])
        valid = os.path.join(args.dataset_dir, config['valid_set'])
        test = os.path.join(args.dataset_dir, config['test_set_name'])  # TODO : need to fix

    train_data_loader = GeneralDataLoader(train, args.nmr_dir,
                                          batch_size=args.batch_size,
                                          protein_sequence_length=args.protein_sequence_length,
                                          chemical_sequence_length=args.chemical_sequence_length,
                                          adapter=get_adapter(args))

    valid_data_loader = GeneralDataLoader(valid, args.nmr_dir,
                                          batch_size=args.batch_size,
                                          protein_sequence_length=args.protein_sequence_length,
                                          chemical_sequence_length=args.chemical_sequence_length,
                                          adapter=get_adapter(args))

    test_data_loader = GeneralDataLoader(test, args.nmr_dir,
                                         batch_size=args.batch_size,
                                         protein_sequence_length=args.protein_sequence_length,
                                         chemical_sequence_length=args.chemical_sequence_length,
                                         adapter=get_adapter(args))

    return train_data_loader, valid_data_loader, test_data_loader


class GeneralDataLoader(object):
    def __init__(self, data_file_name, nmr_dir, batch_size=1,
                 protein_sequence_length=1000,
                 chemical_sequence_length=1000,
                 adapter=lambda x: x,
                 ):
        self.batch_size = batch_size
        self.data_reader = JSONDataReader(data_file_name)
        self.protein_sequence_length = protein_sequence_length
        self.chemical_sequence_length = chemical_sequence_length
        self.adapter = adapter

    def __iter__(self):
        batch = []
        for bind, protein_indices, chemical_indices, nmr_values in self.adapter(self.data_reader):
            if len(protein_indices) < self.protein_sequence_length:
                protein_indices = protein_indices + [0 for x in range(self.protein_sequence_length -
                                                                      len(protein_indices))]
            if len(chemical_indices) < self.protein_sequence_length:
                chemical_indices = chemical_indices + [0 for x in range(self.chemical_sequence_length -
                                                                        len(chemical_indices))]
            batch.append([[bind], protein_indices[:self.protein_sequence_length],
                          chemical_indices[:self.chemical_sequence_length], nmr_values.tolist()])
            if len(batch) == self.batch_size:
                yield [np.array(x) for x in zip(*batch)]
                batch = []
