import json
import os
import xml.etree.ElementTree as elemTree
import pandas as pd

CONFIGURATION_FILE_NAME='configuration.json'


def get_conf():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIGURATION_FILE_NAME)
    with open(file_path) as f:
        return json.load(f)


class XMLManager():
    def __init__(self, filename, keywords, children_tag):
        self.keywords = keywords
        self.xml = None
        self.filename = filename
        self.children_tag = children_tag

    def load_xml_from_cache(self, cache):
        self.xml = cache

    def load_xml(self):
        self.xml = elemTree.parse(self.filename)

    def iter_xml(self, wrapper=lambda x: x):
        path = []
        for event, elem in wrapper(elemTree.iterparse(self.filename, events=("start", "end"))):
            if event == 'start':
                path.append(elem.tag)
            elif event == 'end':
                if elem.tag == self.children_tag and len(path) <= 1: #TODO
                    yield elem

    def element_parser(self, el):
        return {}

    def get_dataframe(self, iter_size=-1):
        output = {k: [] for k in self.keywords}
        for idx, el in enumerate(list(self.xml.getroot())):
            if iter_size > 0 and idx > iter_size: break
            parsed_element = self.element_parser(el)
            for k in self.keywords:
                output[k].append(parsed_element.get(k, None))

        return pd.DataFrame(output)

    def export_to_file(self, objectives, fname):
        output = []
        for el in self.iter_xml():
            parsed_element = self.element_parser(el)
            output.append({k:parsed_element[k] for k in objectives})

        with open(fname) as f:
            for packet in output:
                f.write(json.dumps(packet)+'\n')





