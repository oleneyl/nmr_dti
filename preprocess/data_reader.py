import json
from .kernel import AbstractReader
from .nmr_base import NMRQueryEngine

class LineDataReader(AbstractReader):
    def parse_data(self, line):
        return line.strip()


class JSONDataReader(AbstractReader):
    def parse_data(self, line):
        return json.loads(line.strip())

    @classmethod
    def dump_data(self, datum):
        return json.dumps(datum)


class NMRDataReader(JSONDataReader):
    def __init__(self, fname, nmr_dir):
        super(NMRDataReader, self).__init__(fname)
        self.nmr_query_engine = NMRQueryEngine(nmr_dir)

    def parse_data(self, line):
        meta_info = super(NMRDataReader, self).parse_data(line)
        hmdb_id = meta_info['hmdb_id']
        datum = self.nmr_query_engine.query(hmdb_id)
        if datum != NMRQueryEngine.QUERY_FAIL:
            freq, rg = datum.get_ft()
            meta_info['nmr_freq'] = freq
            meta_info['nmr_rg'] = rg

        return meta_info
