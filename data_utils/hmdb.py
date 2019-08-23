from .kernel import get_conf, XMLManager

def create_hmdb_alignment():
    conf = get_conf()
    hmdb_manager = HMDBXMLManager(conf['hmbd']['xml'])
    hmdb_manager.load_xml()
    hmdb_manager.export_to_file(conf['hmbd']['objectives'], conf['hmdb']['export_endpoint'])

class HMDBXMLManager(XMLManager):
    def __init__(self, filename):
        super(HMDBXMLManager, self).__init__(filename, ['cas_registry_number', 'hmdb_id', 'inchikey', 'protein_associations'], 'metabolite')

    def iter_xml(self, wrapper=lambda x: x):
        super(HMDBXMLManager, self).iter_xml(wrapper=wrapper)

    def element_parser(self, el):
        packet = {
            'cas_registry_number': el.find('{http://www.hmdb.ca}cas_registry_number').text,
            'hmdb_id': [name.text for name in el.find('{http://www.hmdb.ca}secondary_accessions').getchildren()],
            'inchikey': el.find('{http://www.hmdb.ca}inchikey').text,
            'protein_associations': [x.find('{http://www.hmdb.ca}uniprot_id').text for x in
                                     el.find('{http://www.hmdb.ca}protein_associations').getchildren()]
        }
        return packet
