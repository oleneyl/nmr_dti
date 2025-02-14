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
        self.chars = ['[ENDL]', '(', 'I', 'O', '@', '[', '=', 'N', ']', ')', '2',
                      '1', '-', '3', 'H', 'S', '/','B', '.',
                      'P', 'C', '4', '+', '\\', '5', '6', '7', '8', '9', 'l', 'F', '#', 'I', 'r', '#']
        self.reverse_map = {x: idx for idx, x in enumerate(self.chars)}

    def encode(self, sequence):
        return [self.reverse_map[ch] for ch in sequence] + [0]

    def decode(self, indices_array):
        return ''.join([self.chars[idx] for idx in indices_array])


class DTASMILESVocab(Vocab):
    def __init__(self):
        self.reverse_map = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

        self.chars = {self.reverse_map[k]:k for k in self.reverse_map}

    def encode(self, sequence):
        return [self.reverse_map[ch] for ch in sequence] + [0]

    def decode(self, indices_array):
        return ''.join([self.chars[idx] for idx in indices_array])

class DTAProteinVocab(Vocab):
    def __init__(self):
        self.reverse_map = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                             "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                             "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                             "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

        self.chars = {self.reverse_map[k]: k for k in self.reverse_map}

    def encode(self, sequence):
        return [self.reverse_map[ch] for ch in sequence] + [0]

    def decode(self, indices_array):
        return ''.join([self.chars[idx] for idx in indices_array])