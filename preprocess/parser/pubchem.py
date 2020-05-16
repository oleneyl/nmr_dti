import requests
import re
import os
import time
import shutil
import multiprocessing as mp

URI = 'https://pubchem.ncbi.nlm.nih.gov/pc_fetch/pc_fetch.cgi'


def collect_pubchem(start, end):
    # Caution : {end} indexed value will not be included in output
    query = {
        'db': 'pcsubstance',
        'idinput': 'fromstring',
        'idstr': f'{start}-{end}',
        'idfile': '(binary)',
        'retmode': 'smiles',
        'compression': 'gzip',
        'n_conf': 1
    }

    res = requests.post(URI, data=query)
    html_text = res.text
    if re.search(r'(ftp:.*gz)', html_text) is not None:
        ftp_url = re.search(r'(ftp:.*gz)', html_text).group(0)
        reqid = ftp_url.split('/')[-1].split('.')[0]
    else:
        reqid = re.search(r'reqid=([0-9]+)', html_text).group(1)

    return reqid


def check_download_status(reqid):
    status_uri = f'{URI}?reqid={reqid}'
    res = requests.get(status_uri)
    status = re.search(r'<em>(\w+?)</em>', res.text).group(1)
    return status


def get_ftp_address(reqid):
    status_uri = f'{URI}?reqid={reqid}'
    res = requests.get(status_uri)
    ftp_address = re.search(r'(ftp:.*gz)', res.text).group(0).strip()
    return ftp_address


def collect_and_download(start, end, target_dir, polling_interval=1000):
    reqid = collect_pubchem(start, end)
    status = 'Fail'
    while status != 'Done':
        status = check_download_status(reqid)
        time.sleep(polling_interval/1000)
    # Now we can fetch data
    ftp_address = get_ftp_address(reqid)
    fname = '%010d-%010d-pubchem-smiles.txt' % (start, end)
    os.system(f'wget -O "{fname}.gz" "{ftp_address}"')
    # unzip
    os.system(f'gunzip {fname}.gz')
    # Move to data directory
    shutil.move(fname, os.path.join(target_dir, fname))
    return start, end, fname


def collect_many(start, end, target_dir='/DATA/meson324/pubchem', chunk_size=500000, task_pool_size=4):
    task_pool = mp.Pool(processes=task_pool_size)
    tracker = []
    for chunk_start in range(start, end, chunk_size):
        tracker.append(task_pool.apply_async(collect_and_download,
                                             (chunk_start, min(chunk_start+chunk_size, end), target_dir)))
    # Join
    for async_result in tracker:
        print(async_result.get(), 'Done')

if __name__ == '__main__':
    print(collect_pubchem(1, 5000))