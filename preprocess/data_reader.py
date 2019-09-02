import json
from .kernel import AbstractFileReader
from .nmr_base import NMRQueryEngine


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