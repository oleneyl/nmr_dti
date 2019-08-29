from .hmdb import create_hmdb_alignment
from .uniprot import create_uniprot_alignment
from .trainable import join_hmdb_and_uniprot
from .kernel import get_conf, change_configuration

def create_every_alignment(wrapper=None):
    print('Export HMBD database...')
    create_hmdb_alignment(wrapper=wrapper)
    print('Export Uniprot database...')
    create_uniprot_alignment(wrapper=wrapper)

def create_joined_data(wrapper=None):
    conf = get_conf()
    join_hmdb_and_uniprot(conf['hmdb']['export_endpoint'], conf['uniprot']['export_endpoint'], conf['trainable']['export_endpoint'],
                          wrapper=wrapper)