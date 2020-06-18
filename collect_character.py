import string
from preprocess.data_reader import JSONDataReader

readers = [JSONDataReader("/media/DATA/meson324/dataset/ibm3/train"),
JSONDataReader("/media/DATA/meson324/dataset/ibm3/valid"),
JSONDataReader("/media/DATA/meson324/dataset/ibm3/test"),
            JSONDataReader("/media/DATA/meson324/dataset/davis/train"),
            JSONDataReader("/media/DATA/meson324/dataset/kiba3/train"),]

tkns = []

for reader in readers:
    for d in reader:
        smiles = d['smiles']
        for idx, ch in enumerate(smiles):
            tkn = ch
            if tkn in string.ascii_lowercase and idx > 0:
                tkn = smiles[idx-1:idx+1]

            if tkn not in tkns:
                tkns.append(tkn)

    print(tkns)