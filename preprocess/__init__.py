from .hmdb import create_hmdb_alignment
from .uniprot import create_uniprot_alignment
from .trainable import join_hmdb_and_uniprot, create_negatives, mix_nmr_into_hmdb
from .kernel import get_conf, change_configuration
from .data_reader import JSONDataReader
import random

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