import json
from .kernel import AbstractLoader


class LineDataLoader(AbstractLoader):
    def parse_data(self, line):
        return line.strip()


class JSONDataLoader(AbstractLoader):
    def parse_data(self, line):
        return json.loads(line.strip())

    @classmethod
    def dump_data(self, datum):
        return json.dumps(datum)