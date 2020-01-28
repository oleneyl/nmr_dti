"""
Download base file from specific repository.
This will automatically update your configuration file.
"""
import os, sys
import subprocess
from .kernel import get_conf, edit_conf

DEEPDTA_DOWNLOAD_LINK = 'https://drive.google.com/u/0/uc?id=1sfUlAt0H0YSrpDr9nuMlmWLOsn3yMrTx&export=download'
DEEPDTA_DIR_NAME = 'DeepDTA'

IBM_DOWNLOAD_LINK = 'https://drive.google.com/u/0/uc?id=1KG2T0Tl2bqrRXjooE383YrVNYqdbLtPr&export=download'
IBM_DIR_NAME = 'InterpretableDTIP'

def download_file(link, fname):
    command = ['wget', '-O', fname, link]
    ps = subprocess.call(command)


class FileDownloader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _download_and_unzip(self, name, download_link):
        tarname = os.path.join(self.base_dir, name + '.tar.gz')
        dirname = os.path.join(self.base_dir, name)
        download_file(download_link, tarname)
        subprocess.call(['tar', '-xvzf', tarname])
        return dirname

    def download_kiba(self):
        dirname = self._download_and_unzip(DEEPDTA_DIR_NAME, DEEPDTA_DOWNLOAD_LINK)
        edit_conf('kiba', os.path.join(dirname, 'kiba'))
        edit_conf('davis', os.path.join(dirname, 'davis'))

    def download_ibm(self):
        dirname = self._download_and_unzip(DEEPDTA_DIR_NAME, DEEPDTA_DOWNLOAD_LINK)
        edit_conf('ibm', os.path.join(dirname, 'data'))