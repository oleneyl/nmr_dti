import sentencepiece as spm


class Vocab(object):
    """Vocab converts given sequence into list - of - index
    """
    def __init__(self):
        pass

    def __call__(self, sequence):
        return self.encode(sequence)

    def encode(self, sequence):
        raise NotImplementedError

    def decode(self, indices_array):
        raise NotImplementedError


class DummyVocab(Vocab):
    def encode(self, sequence):
        return [0]


class SentencePieceVocab(Vocab):
    def __init__(self, vocab_file):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(vocab_file)

    def encode(self, sequence):
        return self.sp.encode_as_ids(sequence)

    def decode(self, indices_array):
        return self.sp.decode_ids(indices_array)


class SimpleSMILESVocab(Vocab):
    def __init__(self):
        self.chars = ['[ENDL]', '(', 'I', 'O', '@', '[', '=', 'N', ']', ')', '2', '1', '-', '3', 'H', 'S', '/', 'P', 'C', '4', '+', '\\', '5', '6', '7', '8', '9']
        self.reverse_map = {x: idx for idx, x in enumerate(self.chars)}

    def encode(self, sequence):
        return [self.reverse_map[ch] for ch in sequence] + [0]



    def decode(self, indices_array):
        return ''.join([self.chars[idx] for idx in indices_array])