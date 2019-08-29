from .data_reader import JSONDataReader, NMRDataReader
from collections import defaultdict, OrderedDict
import numpy as np


def create_negatives(hmdb_aligned_file, uniprot_aligned_file, size=100, seed=None, wrapper=None):
    hmdb_data = JSONDataReader(hmdb_aligned_file)
    uniprot_data = JSONDataReader(uniprot_aligned_file)

    # Collect information about matching
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
        #chemical = np.random.choice(chemicals, p=sampling_prob)    # This was disabled due to speed issue
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


def mix_nmr_into_hmdb(hmdb_aligned_file, nmr_dir, output_file_name, wrapper = None):
    hmdb_data = NMRDataReader(hmdb_aligned_file, nmr_dir)
    iterator = wrapper(hmdb_data) if wrapper is not None else hmdb_data

    filtered = []
    for item in iterator:
        if 'nmr_freq' in item:
            filtered.append(item)

    print(f'Got chemicals with NMR data : {len(filtered)}')
    JSONDataReader.save_from_raw(filtered, output_file_name)


def join_hmdb_and_uniprot(hmdb_aligned_file, uniprot_aligned_file, wrapper = None):
    '''
    Joining strategy : for each mapping from hmdb [chemical-protein] mapping, yield
    hmdb + protein information preserving each's key-value map.

    :param hmdb_aligned_file:
    :param uniprot_aligned_file:
    :param output_file_name:
    :return:
    '''
    hmdb_data = JSONDataReader(hmdb_aligned_file)
    uniprot_data = JSONDataReader(uniprot_aligned_file)
    query_map = uniprot_data.create_query_map('uniprot_id')

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