try:
    import nmrglue as ng
except:
    raise ImportError("Failed import model nmrglue; Check your python env has nmrglue package.")

import os
import zipfile
from tqdm import tqdm
from collections import defaultdict

class NMRDatum(object):
    LOAD_FAIL = 1

    def __init__(self, dic, data, dtype, dim=1):
        self._dic = dic
        self._data = data
        self.dtype = dtype
        self.dim = dim
        self.udic = {}

    def help_udic(self):
        udic = {}
        if self.dtype == 'varian':
            udic['sw'] = float(self._dic['procpar']['sw']['values'][0])
            udic['obs'] = float(self._dic['procpar']['sfrq']['values'][0])
            udic['car'] = (-float(self._dic['procpar']['reffrq']['values'][0]) + float(
                self._dic['procpar']['sfrq']['values'][0])) * 1000 * 1000

        return udic

    def unit_conversion(self):
        if self.dtype == 'varian':
            udic = self.help_udic()
            uc = ng.fileiobase.unit_conversion(self._data.shape[0], True, udic['sw'], udic['obs'], udic['car'])
        elif self.dtype == 'bruker':
            udic = ng.bruker.guess_udic(self._dic, self._data)
            uc = ng.fileiobase.unit_conversion(self._data.shape[0], True, udic[0]['sw'], udic[0]['obs'], udic[0]['car'])

        return uc

    def get_ft(self):
        data = ng.proc_base.fft(self._data)
        data = ng.proc_base.ps(data, p0=0.0)  # phase correction
        data = ng.proc_base.abs(data)
        data = ng.proc_base.di(data)  # discard the imaginaries
        if self.dtype == 'bruker':
            data = list(reversed(data))

        uc = self.unit_conversion()
        return data, (uc.ppm(0), uc.ppm(len(data)))

    @classmethod
    def load(self, dirpath, ignore_exception=True):
        if 'oned' in dirpath:
            dim = 1
        else:
            dim = 2
        # Guess which type of input was given
        try:
            if os.path.exists(os.path.join(dirpath, 'acqu')):
                dic, data = ng.bruker.read(dirpath, read_procs=False)
                dtype = 'bruker'
            elif os.path.exists(os.path.join(dirpath, 'fid')):
                dic, data = ng.varian.read(os.path.join(dirpath))
                dtype = 'varian'
            elif os.path.exists(os.path.join(dirpath, 'FID')):
                dic, data = ng.varian.read(os.path.join(dirpath), fid_file="FID", procpar_file="PROCPAR")
                dtype = 'varian'
            elif os.path.exists(os.path.join(dirpath, 'fid.gif')):  # Some very bad people do bad things
                dic, data = ng.varian.read(os.path.join(dirpath), fid_file="fid.gif", procpar_file="procpar")
                dtype = 'varian'
            else:
                try:
                    dic, data = ng.sparky.read(self.path)
                except:
                    try:
                        dic, data = ng.pipe.read(self.path)
                    except:
                        raise
        except Exception as e:
            try:
                dic, data = ng.bruker.read(dirpath, read_procs=False)
                dtype = 'bruker'
            except:
                if ignore_exception:
                    return self.LOAD_FAIL
                else:
                    raise
        return NMRDatum(dic, data, dtype, dim)

    def spectrum(self):
        return self._data

    def unzip(self):
        return self._dic, self._data, self.dtype, self.dim


class NMRQueryEngine(object):
    QUERY_FAIL = 0

    ## Loads unzipped directory
    def __init__(self, dirname):
        self.dirname = dirname
        self.dirs = os.listdir(dirname)

    @classmethod
    def extract(self, origin_dir, target_dir):
        fnames = os.listdir(origin_dir)
        for fname in tqdm(fnames):
            with zipfile.ZipFile(os.path.join(origin_dir, fname), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(target_dir, fname))

    def query(self, hmdb_id, ignore_exception=True):
        target_path = None
        for path in self.dirs:
            if int(path.split('_')[0][4:]) == hmdb_id:
                if 'oned' in path:
                    target_path = path
                    break

        if target_path is None:  # Query fail
            return self.QUERY_FAIL
        else:
            log_dir_cands = ['__MACOSX']
            target_path = os.path.join(self.dirname, target_path)
            # get innermost directory
            while len([x for x in os.listdir(target_path) if x not in log_dir_cands]) == 1:
                target_path = os.path.join(target_path, os.listdir(target_path)[0])

        return NMRDatum.load(target_path, ignore_exception=ignore_exception)

    def all(self):
        status = defaultdict(list)
        for path in self.dirs:
            try:
                hmdb_id = int(path.split('_')[0][4:])
                output = self.query(hmdb_id)
                if output == NMRDatum.LOAD_FAIL:
                    status['fail'].append(hmdb_id)
                else:
                    status[output.dtype].append(hmdb_id)
            except:
                print(hmdb_id)
                raise

        return status