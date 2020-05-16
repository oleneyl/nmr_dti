from preprocess.kernel import get_conf, XMLManager


def create_hmdb_alignment(wrapper = None):
    conf = get_conf()
    hmdb_manager = HMDBXMLManager(conf['hmdb']['xml'])
    hmdb_manager.export_to_file(conf['hmdb']['objectives'], conf['hmdb']['export_endpoint'], wrapper=wrapper)


class HMDBXMLManager(XMLManager):
    def __init__(self, filename):
        super(HMDBXMLManager, self).__init__(filename, ['cas_registry_number', 'hmdb_id', 'inchikey', 'protein_associations','smiles',"pubchem_id"], '{http://www.hmdb.ca}metabolite')

    def iter_xml(self, wrapper=lambda x: x):
        for el in super(HMDBXMLManager, self).iter_xml(wrapper=wrapper):
            yield el

    def element_parser(self, el):
        packet = {
            'cas_registry_number': el.find('{http://www.hmdb.ca}cas_registry_number').text,
            'hmdb_id': [name.text for name in el.find('{http://www.hmdb.ca}secondary_accessions').getchildren()],
            'inchikey': el.find('{http://www.hmdb.ca}inchikey').text,
            'protein_associations': [x.find('{http://www.hmdb.ca}uniprot_id').text for x in
                                     el.find('{http://www.hmdb.ca}protein_associations').getchildren()],
            'smiles': el.find('{http://www.hmdb.ca}smiles').text,
            'pubchem_id': el.find('{http://www.hmdb.ca}pubchem_compound_id').text
        }
        return packet
