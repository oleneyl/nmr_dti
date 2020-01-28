"""
Download base file from specific repository.
This will automatically update your configuration file.
"""
import os, sys
import subprocess
from .kernel import get_conf, edit_conf

DEEPDTA_DOWNLOAD_LINK = 'https://drive.google.com/u/0/uc?id=1sfUlAt0H0YSrpDr9nuMlmWLOsn3yMrTx&export=download'
DEEPDTA_DIR_NAME = 'DeepDTA'


def download_file(link, fname):
    command = ['wget', '-O', fname, link]
    ps = subprocess.call(command)


class FileDownloader:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def download_kiba(self):
        tarname = os.path.join(self.base_dir, DEEPDTA_DIR_NAME + '.tar.gz')
        dirname = os.path.join(self.base_dir, DEEPDTA_DIR_NAME)
        download_file(DEEPDTA_DOWNLOAD_LINK, tarname)
        subprocess.call(['tar', '-xvzf', tarname])
        edit_conf('kiba', os.path.join(dirname, 'kiba'))
        edit_conf('davis', os.path.join(dirname, 'davis'))


if __name__ == '__main__':
    downloader = FileDownloader('.')
    downloader.download_kiba()