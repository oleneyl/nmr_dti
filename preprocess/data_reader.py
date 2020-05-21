import json
import pickle
from .kernel import AbstractFileReader
from preprocess.parser.nmr_base import NMRQueryEngine


class LineDataReader(AbstractFileReader):
    def parse_data(self, line):
        return line.strip()


class JSONDataReader(AbstractFileReader):
    def parse_data(self, line):
        return json.loads(line.strip())

    @classmethod
    def dump_data(self, datum):
        return json.dumps(datum)


class NMRDataReader(JSONDataReader):
    def __init__(self, fname, nmr_dir):
        super(NMRDataReader, self).__init__(fname)
        self.nmr_query_engine = NMRQueryEngine(nmr_dir)

    def parse_data(self, line, dim=1):
        meta_info = super(NMRDataReader, self).parse_data(line)
        hmdb_id = meta_info['hmdb_id']
        if len(hmdb_id) == 0:
            return meta_info
        datum = self.nmr_query_engine.query(int(hmdb_id[0][4:]), dim=dim)
        if datum != NMRQueryEngine.QUERY_FAIL and len(datum.shape()) == dim:
            freq, rg = datum.get_ft()
            meta_info['nmr_freq'] = freq
            meta_info['nmr_rg'] = rg

        return meta_info


class RestrictiveNMRDataReader(NMRDataReader):
    def __iter__(self):
        with open(self.fname) as f:
            for line in f:
                output = self.parse_data(line)
                if 'nmr_freg' not in output:
                    continue
                else:
                    yield output


class DataFrameReader(AbstractFileReader):
    def __init__(self, fname):
        self.fname = fname
        with open(fname, 'rb') as f:
            self._cache = pickle.load(f)

    @classmethod
    def dump_data(cls, datum):
        return datum

    @classmethod
    def save_from_raw(cls, data, fname):
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    def __iter__(self):
        if len(self._cache) > 0:
            for idx in range(len(self._cache)):
                yield self[idx]
        else:
            raise IndexError("NMRPredictionDatasetReader must be cached")

    def __len__(self):
        return len(self._cache)

    def cache_data(self):
        self._cache = list(self)

    def __getitem__(self, idx):
        return self._cache.iloc[idx]


class NMRDataFrameReader(DataFrameReader):
    def __getitem__(self, idx):
        return self._cache[idx]
