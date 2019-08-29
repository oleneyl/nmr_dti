from ..preprocess.data_reader import JSONDataReader
from .numpy_adapter import get_adapter


def data_loader_args(parser):
    group = parser.add_group('data_loader')
    group.add_argument('--data_file_name', type=str)
    group.add_argument('--batch_size', type=str)
    group.add_argument('--protein_sequence_length', type=int)


def get_data_loader(args):
    return GeneralDataLoader(args.data_file_name,
                             batch_size=args.batch_size,
                             protein_sequence_length=args.protein_sequence_length,
                             adapter=get_adapter(args))


class GeneralDataLoader(object):
    def __init__(self, data_file_name, batch_size=1,
                 protein_sequence_length = 1000,
                 adapter = lambda x: x
                 ):
        self.batch_size = batch_size
        self.data_reader = JSONDataReader(data_file_name)
        self.protein_sequence_length = protein_sequence_length
        self.adapters = adapter

    def __iter__(self):
        batch = []
        for bind, protein_indices, chemical_indices, nmr_values in self.adapter(self.data_reader):
            if len(batch) < self.batch_size:
                batch.append([bind, protein_indices, chemical_indices, nmr_values])
            else:
                yield list(zip(*batch))
