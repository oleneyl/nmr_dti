from .data_reader import NMRDataReader, JSONDataReader
from .kernel import BaseIterable
from collections import defaultdict, OrderedDict
import numpy as np


def create_positive_dataset(hmdb_data, uniprot_data, wrapper=None):
    query_map = BaseIterable(uniprot_data).create_query_map('uniprot_id')

    iterator = wrapper(hmdb_data) if wrapper is not None else hmdb_data

    joined_data = []
    for datum in iterator:
        for uniprot_id in datum['protein_associations']:
            if uniprot_id in query_map:
                output = {}
                output.update(datum)
                output.update(query_map[uniprot_id])
                output['bind'] = True
                joined_data.append(output)

    return joined_data


def create_negative_dataset(hmdb_data, uniprot_data, size=1, wrapper=lambda x: x):
    """
    Create dataset with given negative portion
    """
    chemicals = [x for x in hmdb_data]
    chemical_size = len(chemicals)

    avail_proteins = []
    for chem in chemicals:
        avail_proteins += chem['protein_associations']
    avail_proteins = list(set(avail_proteins))
    print('Available protein extraction done')

    proteins = {x['uniprot_id']: x for x in wrapper(uniprot_data) if x['uniprot_id'] in avail_proteins}
    protein_indices = [x for x in proteins]
    protein_length = len(proteins)
    print(f'Protein map generation done : {protein_length}')
    sampling_prob = [protein_length - len(chem['protein_associations']) for chem in chemicals]
    sampling_prob = np.array(sampling_prob) / sum(sampling_prob)
    output = []

    for i in (range(size) if wrapper is None else wrapper(range(size))):
        # chemical = np.random.choice(chemicals, p=sampling_prob)    # This was disabled due to speed issue
        chemical = chemicals[np.random.randint(0, chemical_size)]
        repeat_max = 20
        protein_id = '!NOTASSIGNED'
        while (repeat_max == 20 or protein_id in chemical['protein_associations']) and repeat_max > 0:
            index = np.random.randint(0, protein_length)
            protein_id = protein_indices[index]
            repeat_max -= 1
        if repeat_max > 0:
            datum = {}
            datum.update(chemical)
            datum.update(proteins[protein_id])
            datum['bind'] = False
            output.append(datum)
        else:
            raise ValueError('Too many iteration occurs')

    return output


def create_dataset(hmdb_data, uniprot_data, negative_ratio=0.0, wrapper = None):
    positives = create_positive_dataset(hmdb_data, uniprot_data, wrapper=wrapper)
    negatives = create_negative_dataset(hmdb_data, uniprot_data,
                                        size=int(len(positives)*negative_ratio), wrapper=wrapper)

    return positives, negatives


def strict_splitting(hmdb_data, uniprot_data, split_ratio=0.5, wrapper=None):
    hmdb_data = list(hmdb_data)
    uniprot_data = list(uniprot_data)

    split_point = int(len(hmdb_data)*split_ratio)
    upper_chemical = hmdb_data[:split_point]
    lower_chemical = hmdb_data[split_point:]

    protein_over_upper = []
    for datum in upper_chemical:
        protein_over_upper += datum['protein_associations']

    protein_over_lower = []
    for datum in lower_chemical:
        protein_over_lower += datum['protein_associations']

    protein_over_upper = set(protein_over_upper)
    protein_over_lower = set(protein_over_lower)
    protein_over_lower = protein_over_lower - protein_over_upper

    upper_protein = [x for x in uniprot_data if x['uniprot_id'] in protein_over_upper]
    lower_protein = [x for x in uniprot_data if x['uniprot_id'] in protein_over_lower]

    return (upper_chemical, upper_protein), (lower_chemical, lower_protein)


def create_negatives(hmdb_aligned_file, uniprot_aligned_file, size=100, seed=None, wrapper=None):
    print('Using deprecated function create_negatives:: Please use create_negative_dataset instead.')
    hmdb_data = JSONDataReader(hmdb_aligned_file)
    uniprot_data = JSONDataReader(uniprot_aligned_file)

    output = create_negative_dataset(hmdb_data, uniprot_data)

    return output


def mix_nmr_into_hmdb(hmdb_aligned_file, nmr_dir, output_file_name, wrapper = None):
    hmdb_data = NMRDataReader(hmdb_aligned_file, nmr_dir)
    iterator = wrapper(hmdb_data) if wrapper is not None else hmdb_data

    filtered = []
    for item in iterator:
        if 'nmr_freq' in item:
            filtered.append(item)

    print(f'Got chemicals with NMR data : {len(filtered)}')
    JSONDataReader.save_from_raw(filtered, output_file_name)


def join_hmdb_and_uniprot(hmdb_aligned_file, uniprot_aligned_file, wrapper=None):
    """
    Joining strategy : for each mapping from hmdb [chemical-protein] mapping, yield
    hmdb + protein information preserving each's key-value map.

    :param hmdb_aligned_file:
    :param uniprot_aligned_file:
    :return:
    """
    print('Using deprecated join_hmdb_and_uniport::use create_positive_dataset instead.')
    hmdb_data = JSONDataReader(hmdb_aligned_file)
    uniprot_data = JSONDataReader(uniprot_aligned_file)

    return create_positive_dataset(hmdb_data, uniprot_data)