import json
import os
import xml.etree.ElementTree as elemTree
import pandas as pd

CONFIGURATION_FILE_NAME = 'configuration.json'


def change_configuration(conf_file_name):
    global CONFIGURATION_FILE_NAME
    CONFIGURATION_FILE_NAME = conf_file_name


def get_conf_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIGURATION_FILE_NAME)


def get_conf():
    file_path = get_conf_file()
    with open(file_path) as f:
        return json.load(f)


def edit_conf(dict_opt):
    """
    This function will PERMANANTLY change referring configuration.
    This function only be used when downloading raw file.
    """
    current_conf = get_conf()
    for kwd in dict_opt:
        current_conf[kwd] = dict_opt[kwd]
    with open(get_conf_file(), 'w') as f:
        json.dump(current_conf, f, indent=4, ensure_ascii=False)


class XMLManager(object):
    def __init__(self, filename, keywords, children_tag):
        self.keywords = keywords
        self.xml = None
        self.filename = filename
        self.children_tag = children_tag

    def load_xml_from_cache(self, cache):
        self.xml = cache

    def load_xml(self):
        self.xml = elemTree.parse(self.filename)

    def iter_xml(self, wrapper=None):
        path = []
        iterator = elemTree.iterparse(self.filename, events=("start", "end"))
        if wrapper is not None:
            iterator = wrapper(iterator)

        for event, elem in iterator:
            if event == 'start':
                path.append(elem.tag)
            elif event == 'end':
                if elem.tag == self.children_tag and len(path) <= 2:
                    yield elem
                path.pop()

    def element_parser(self, el):
        return {}

    def get_dataframe(self, iter_size=-1):
        output = {k: [] for k in self.keywords}
        for idx, el in enumerate(list(self.xml.getroot())):
            if idx > iter_size > 0:
                break
            parsed_element = self.element_parser(el)
            for k in self.keywords:
                output[k].append(parsed_element.get(k, None))

        return pd.DataFrame(output)

    def export_to_file(self, objectives, fname, wrapper=None):
        output = []
        for el in self.iter_xml(wrapper=wrapper):
            parsed_element = self.element_parser(el)
            output.append({k: parsed_element[k] for k in objectives})

        print(f'Total input {len(output)} detected')
        with open(fname, 'w') as f:
            for packet in output:
                f.write(json.dumps(packet)+'\n')


class AbstractIterable(object):
    def create_query_map(self, primary_key):
        output = {}
        for datum in self:
            output[datum[primary_key]] = datum

        return output


class BaseIterable(AbstractIterable):
    def __init__(self, iterable):
        self._iterable = iterable
        self._cache = []

    def __iter__(self):
        for x in self._iterable:
            yield x


class AbstractFileReader(AbstractIterable):
    def __init__(self, fname):
        self.fname = fname
        self._cache = []

    def parse_data(self, line):
        return line

    @classmethod
    def dump_data(cls, datum):
        return datum

    @classmethod
    def save_from_raw(cls, data, fname):
        with open(fname, 'w') as f:
            for datum in data:
                try:
                    f.write(cls.dump_data(datum) + '\n')
                except Exception as e:
                    print(datum)
                    raise

    @classmethod
    def save_train_valid_test(cls, train, valid, test, save_dir):
        """
        Shortcut for train-valid-test tuple dataset saving
        """
        cls.save_from_raw(train, os.path.join(save_dir, 'train'))
        cls.save_from_raw(valid, os.path.join(save_dir, 'valid'))
        cls.save_from_raw(test, os.path.join(save_dir, 'test'))

    def __iter__(self):
        if len(self._cache) > 0:
            for obj in self._cache:
                yield obj
        with open(self.fname) as f:
            for line in f:
                yield self.parse_data(line)

    def cache_data(self):
        self._cache = list(self)