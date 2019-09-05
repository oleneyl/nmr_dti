from ..data_reader import JSONDataReader, NMRDataReader, RestrictiveNMRDataReader
from ..kernel import get_conf
from .numpy_adapter import get_adapter
import numpy as np


def data_loader_args(parser):
    base_conf = get_conf()
    group = parser.add_argument_group('data_loader')
    group.add_argument('--train_file_name', type=str, default=base_conf['trainable']['train'])
    group.add_argument('--valid_file_name', type=str, default=base_conf['trainable']['valid'])
    group.add_argument('--nmr_dir', type=str)
    group.add_argument('--batch_size', type=int)
    group.add_argument('--protein_sequence_length', type=int, default=256)
    group.add_argument('--chemical_sequence_length', type=int, default=128)


def get_data_loader(args):
    train_data_loader = GeneralDataLoader(args.train_file_name, args.nmr_dir,
                                          batch_size=args.batch_size,
                                          protein_sequence_length=args.protein_sequence_length,
                                          chemical_sequence_length=args.chemical_sequence_length,
                                          adapter=get_adapter(args))

    valid_data_loader = GeneralDataLoader(args.valid_file_name, args.nmr_dir,
                                          batch_size=args.batch_size,
                                          protein_sequence_length=args.protein_sequence_length,
                                          chemical_sequence_length=args.chemical_sequence_length,
                                          adapter=get_adapter(args))

    return train_data_loader, valid_data_loader


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
