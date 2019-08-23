from .hmdb import create_hmdb_alignment
from .uniprot import create_uniprot_alignment

def create_every_alignment():
    create_hmdb_alignment()
    create_uniprot_alignment()