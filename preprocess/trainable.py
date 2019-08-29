from .data_reader import JSONDataReader

def join_hmdb_and_uniprot(hmdb_aligned_file, uniprot_aligned_file, output_file_name, wrapper = None):
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
                joined_data.append(output)

    JSONDataReader.save_from_raw(joined_data, output_file_name)