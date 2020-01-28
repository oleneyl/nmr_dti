from .hmdb import create_hmdb_alignment
from .uniprot import create_uniprot_alignment
from .tool import join_hmdb_and_uniprot, create_negatives, mix_nmr_into_hmdb
from .tool import strict_splitting, create_dataset
from .kernel import get_conf, change_configuration
from .data_reader import JSONDataReader
from preprocess.processor.ibm_dataset import get_ibm_data_reader
from preprocess.processor.kiba import get_kiba_dataset, get_davis_dataset

from .data_conf import add_data_config
import random
import os


def mix_nmr(output_file_name, wrapper=None):
    conf = get_conf()
    mix_nmr_into_hmdb(conf['hmdb']['export_endpoint'], conf['nmr']['engine_dir'], output_file_name,
                      wrapper=wrapper)


def create_every_alignment(wrapper=None):
    print('Export HMBD database...')
    create_hmdb_alignment(wrapper=wrapper)
    print('Export Uniprot database...')
    create_uniprot_alignment(wrapper=wrapper)


def shuffle_train_valid_split(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    split_point = int(data_length * split_ratio)
    return data[:split_point], data[split_point:]


def mix_positive_and_negative(pos, neg):
    """
    Shuffle positive data and negative data.
    """
    li = pos + neg
    random.shuffle(li)
    return li


def strict_split_data(split_ratio, save_dir, wrapper=None, use_nmr=False):
    """
    Create dataset as splitted <strictly>, which ensure given datasets do not have any overlap
    of protein or chemical.
    """
    os.makedirs(save_dir, exist_ok=True)
    conf = get_conf()
    hmdb_file = conf['hmdb']['nmr_endpoint'] if use_nmr else conf['hmdb']['export_endpoint']
    hmdb_data = JSONDataReader(hmdb_file)
    uniprot_data = JSONDataReader(conf['uniprot']['export_endpoint'])

    # Strict splitting
    print('Start strict splitting..')
    train_tuple, valid_tuple = strict_splitting(hmdb_data, uniprot_data, split_ratio=split_ratio, wrapper=wrapper)
    print('Strict splitting done!')

    # Mix positive and negative(not-positive) data inside
    print(f'Train chemicals {len(train_tuple[0])}, Validation chemicals {len(valid_tuple[0])}')
    train_set = mix_positive_and_negative(*create_dataset(*train_tuple, negative_ratio=1, wrapper=wrapper))
    valid_set = mix_positive_and_negative(*create_dataset(*valid_tuple, negative_ratio=1, wrapper=wrapper))
    print(f'Training set {len(train_set)}, Validation set {len(valid_set)}')

    # Save dataset
    JSONDataReader.save_from_raw(train_set, os.path.join(save_dir, 'train'))
    JSONDataReader.save_from_raw(valid_set, os.path.join(save_dir, 'valid'))
    add_data_config(save_dir, train_set_name='train', valid_set_name='valid',
                    data_read_type='line_json', origin='HMDB', includes_nmr=use_nmr)


def create_dataset_from_ibm(save_dir, wrapper=None):
    # This action do not includes any NMR data inside dataset..
    # This dataset will only for reproduction / baseline check
    os.makedirs(save_dir, exist_ok=True)
    train_reader, valid_reader, test_reader = get_ibm_data_reader()
    train_set = mix_positive_and_negative(*train_reader.create_dataset())
    valid_set = mix_positive_and_negative(*valid_reader.create_dataset())
    test_set = mix_positive_and_negative(*test_reader.create_dataset())

    JSONDataReader.save_train_valid_test(train_set, valid_set, test_set, save_dir)

    add_data_config(save_dir, train_set_name='train', valid_set_name='valid',
                    data_read_type='line_json', origin='IBM', includes_nmr=False, test_set_name='test')


def create_dataset_from_kiba(save_dir, wrapper=None, as_binary=False):
    """
    Create dataset from KIBA
    file must be downloaded and path must be specified in configuration.
    """
    train, valid, test = get_kiba_dataset(get_conf()['kiba'], as_binary=as_binary)
    # No negative data mixing needed
    os.makedirs(save_dir, exist_ok=True)

    JSONDataReader.save_train_valid_test(train, valid, test, save_dir)

    add_data_config(save_dir, train_set_name='train', valid_set_name='valid',
                    data_read_type='line_json', origin='kiba', includes_nmr=False, test_set_name='test',
                    as_binary=as_binary)


def create_dataset_from_davis(save_dir, wrapper=None, as_binary=False):
    """
    Create dataset from DAVIS
    file must be downloaded and path must be specified in configuration.
    """
    train, valid, test = get_davis_dataset(get_conf()['davis'], as_binary=as_binary)
    # No negative data mixing needed
    os.makedirs(save_dir, exist_ok=True)

    JSONDataReader.save_from_raw(train, os.path.join(save_dir, 'train'))
    JSONDataReader.save_from_raw(valid, os.path.join(save_dir, 'valid'))
    JSONDataReader.save_from_raw(test, os.path.join(save_dir, 'test'))

    add_data_config(save_dir, train_set_name='train', valid_set_name='valid',
                    data_read_type='line_json', origin='davis', includes_nmr=False, test_set_name='test',
                    as_binary=as_binary)
