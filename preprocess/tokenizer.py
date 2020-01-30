import sentencepiece as spm
from .kernel import get_conf
from .data_reader import JSONDataReader
import os
TEMP_FILE_NAME = '.temp-tknyze-nmr_infer'


def create_protein_vocab(output_prefix):
    #Create protein vocabulary
    conf = get_conf()
    proteins = JSONDataReader(conf['uniprot']['export_endpoint'])
    with open(TEMP_FILE_NAME, 'w') as f:
        for protein in proteins:
            f.write(protein['sequence'].strip()+'\n')

    spm.SentencePieceTrainer.Train(f'--input={TEMP_FILE_NAME} --model_prefix={output_prefix} --vocab_size=1000')


def create_chemical_vocab(output_prefix):
    sequences = []
    #Create chemical vocabulary
    conf = get_conf()
    INPUT='/DATA/meson324/InterpretableDTIP/data/train/chem.repr'
    spm.SentencePieceTrainer.Train(f'--input={INPUT} --model_prefix={output_prefix} --vocab_size=1000 --hard_vocab_limit=false')


def create_chemical_vocab_from_dataset(dataset_file, output_prefix):
    data_reader = JSONDataReader(dataset_file)
    chemcial_collection_file = dataset_file + '.chemicals'
    with open(chemcial_collection_file, 'w') as f:
        for datum in data_reader:
            f.write(datum['smiles'] + '\n')

    #Create chemical vocabulary
    INPUT=chemcial_collection_file
    spm.SentencePieceTrainer.Train(f'--input={INPUT} --model_prefix={output_prefix} --vocab_size=1000 --hard_vocab_limit=false')