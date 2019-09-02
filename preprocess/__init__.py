from .hmdb import create_hmdb_alignment
from .uniprot import create_uniprot_alignment
from .trainable import join_hmdb_and_uniprot, create_negatives, mix_nmr_into_hmdb
from .trainable import strict_splitting, create_dataset
from .kernel import get_conf, change_configuration
from .data_reader import JSONDataReader
import random
import os


def mix_nmr(output_file_name, wrapper = None):
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


def create_trainable_data(split_ratio, wrapper=None, use_nmr=False):
    conf = get_conf()
    hmdb_file = conf['hmdb']['nmr_endpoint'] if use_nmr else conf['hmdb']['export_endpoint']
    positives = join_hmdb_and_uniprot(hmdb_file, conf['uniprot']['export_endpoint'],
                                      wrapper=wrapper)
    negatives = create_negatives(hmdb_file, conf['uniprot']['export_endpoint'],
                                 size=len(positives), wrapper=wrapper)

    train_data = positives + negatives

    train, valid = shuffle_train_valid_split(train_data, split_ratio)
    JSONDataReader.save_from_raw(train, conf['trainable']['train'])
    JSONDataReader.save_from_raw(valid, conf['trainable']['valid'])


def strict_split_data(split_ratio, save_dir, wrapper=None, use_nmr=False):
    os.makedirs(save_dir, exist_ok=True)
    conf = get_conf()
    hmdb_file = conf['hmdb']['nmr_endpoint'] if use_nmr else conf['hmdb']['export_endpoint']
    hmdb_data = JSONDataReader(hmdb_file)
    uniprot_data = JSONDataReader(conf['uniprot']['export_endpoint'])

    print('Start strict splitting..')
    train_tuple, valid_tuple = strict_splitting(hmdb_data, uniprot_data, split_ratio=split_ratio, wrapper=wrapper)
    print('Strict splitting done!')
    print(f'Train chemicals {len(train_tuple[0])}, Validation chemicals {len(valid_tuple[0])}')
    train_set = create_dataset(*train_tuple, negative_ratio=0.5, wrapper=wrapper)
    valid_set = create_dataset(*valid_tuple, negative_ratio=0.5, wrapper=wrapper)
    train_set = train_set[0] + train_set[1]
    valid_set = valid_set[0] + valid_set[1]
    random.shuffle(train_set)
    random.shuffle(valid_set)
    print(f'Training set {len(train_set)}, Validation set {len(valid_set)}')

    JSONDataReader.save_from_raw(train_set, os.path.join(save_dir, 'train'))
    JSONDataReader.save_from_raw(valid_set, os.path.join(save_dir, 'valid'))
