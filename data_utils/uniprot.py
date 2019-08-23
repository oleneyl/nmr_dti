from .kernel import get_conf, XMLManager

def create_uniprot_alignment():
    conf = get_conf()
    uniprot_manager = UniprotXMLManager(conf['uniprot']['xml'])
    uniprot_manager.load_xml()
    uniprot_manager.export_to_file(conf['uniprot']['objectives'], conf['uniprot']['export_endpoint'])


class UniprotXMLManager(XMLManager):
    def __init__(self, filename):
        super(UniprotXMLManager, self).__init__(filename, ['uniprot_id', 'subcellular'], 'drug')

    def element_parser(self, el):
        subcellular = el.findall('{http://uniprot.org/uniprot}subcellularLocation')
        subcellular = [[x.text for x in e.getchildren()] for e in subcellular]
        uniprot_id = el.find('{http://uniprot.org/uniprot}accession').text
        sequence = el.find('{http://uniprot.org/uniprot}sequence').text
        packet = {
            'subcellular': subcellular,
            'uniprot_id' : uniprot_id,
            'sequence' : sequence
        }
        return packet

    def iter_xml(self, wrapper=lambda x: x):
        super(UniprotXMLManager, self).iter_xml(wrapper=wrapper)

    def query(self, uniprot_id):
        for el in self.iter_xml():
            if el.find('{http://uniprot.org/uniprot}accession') == uniprot_id:
                return el

        return None

    def query_multiple(self, uniprot_ids):
        output = []
        for el in self.iter_xml():
            if el.find('{http://uniprot.org/uniprot}accession').text in uniprot_ids:
                output.append(el)

        return output