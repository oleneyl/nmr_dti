import sentencepiece as spm
from .kernel import get_conf
from .data_reader import JSONDataReader

TEMP_FILE_NAME = '.temp-tknyze-nmr_infer'

def create_protein_vocab(output_prefix):
    sequences = []
    #Create protein vocabulary
    conf = get_conf()
    proteins = JSONDataReader(conf['uniprot']['export_endpoint'])
    with open(TEMP_FILE_NAME, 'w') as f:
        for protein in proteins:
            f.write(protein['sequence'].strip()+'\n')

    spm.SentencePieceTrainer.Train(f'--input={TEMP_FILE_NAME} --model_prefix={output_prefix} --vocab_size=1000')