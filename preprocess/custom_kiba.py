import numpy as np
import os
from collections import OrderedDict

def collect_from_raw_file(dir_name, seperator = '\t'):
    NOT_A_NUMBER = np.nan
    arr = []
    
    file_name = os.path.join(dir_name, 'kiba_binding_affinity_v2.txt')
    with open(file_name) as f:
        for line in f:
            arr.append([(float(x) if x != 'nan' else NOT_A_NUMBER) else  for x in line.strip().split(seperator)])
   
    PROTEIN_FILE_NAME = 'proteins.txt'
    LIGAND_FILE_NAME = 'ligands_can.txt'
    BINARY_CRITERION = 12.1  # https://arxiv.org/pdf/1908.06760.pdf page 10

    with open(os.path.join(dir_path, PROTEIN_FILE_NAME)) as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
        proteins_key = [x for x in proteins]

    with open(os.path.join(dir_path, LIGAND_FILE_NAME)) as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
        ligands_key = [x for x in ligands]

    arr = np.array(arr, type=np.float)
    r_index, c_index = np.where(np.isnan(arr) == False)
    

def parser_from_deep_dta(fpath):
    ligands = json.load(open(os.path.join(fpath,"ligands_can.txt")), object_pairs_hook=OrderedDict)
    proteins = json.load(open(os.path,join(fpath,"proteins.txt")), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(fpath + "Y","rb"), encoding='latin1') ### TODO: read from raw
    if FLAGS.is_log:
        Y = -(np.log10(Y/(math.pow(10,9))))

    XD = []
    XT = []

    if with_label:
        for d in ligands.keys():
            XD.append(label_smiles(ligands[d], self.SMILEN, self.charsmiset))

        for t in proteins.keys():
            XT.append(label_sequence(proteins[t], self.SEQLEN, self.charseqset))
    