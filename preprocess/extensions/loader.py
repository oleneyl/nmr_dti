import deepchem
import os
import numpy as np

DEFAULT_DIR = deepchem.utils.get_data_dir()
GDB9_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
QM9_CSV_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/qm9.csv'


class CoordinateFeaturizer(object):
    def featurize(self, mols, verbose=True, log_every_n=1000):
        """
        Calculate features for molecules.

        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        """

        mols = list(mols)
        features = []

        for i, mol in enumerate(mols):
            if mol is not None:
                features.append(self._featurize(mol))
            else:
                features.append(np.array([]))

        features = np.asarray(features)

        return features

    def _featurize(self, mol):
        """
        Calculate features for a single molecule.
        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        # mol = mol.addHs()
        return mol, mol.GetConformers()[0].GetPositions()

    def __call__(self, mols):
        """
        Calculate features for molecules.
        Parameters
        ----------
        mols : iterable
            RDKit Mol objects.
        """
        return self.featurize(mols)

    def __repr__(self):
        return 'coord_feature'


def load_qm9(featurizer=CoordinateFeaturizer(),
             split='random',
             reload=True,
             move_mean=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
    """Load qm9 datasets."""
    # Featurize qm9 dataset
    qm9_tasks = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298",
        "h298", "g298"
    ]
    if data_dir is None:
        data_dir = DEFAULT_DIR

    if save_dir is None:
        save_dir = DEFAULT_DIR

    if reload:
        save_folder = os.path.join(save_dir, "qm9-featurized")
        if not move_mean:
            save_folder = os.path.join(save_folder, str(featurizer) + "_mean_unmoved")
        else:
            save_folder = os.path.join(save_folder, str(featurizer))

        save_folder = os.path.join(save_folder, str(split))
        loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
            save_folder)

        if loaded:
            return qm9_tasks, all_dataset, transformers

    print('*** reading... ***')

    dataset_file = os.path.join(data_dir, "gdb9.sdf")
    if not os.path.exists(dataset_file):
        deepchem.utils.download_url(url=GDB9_URL, dest_dir=data_dir)
        deepchem.utils.untargz_file(
            os.path.join(data_dir, 'gdb9.tar.gz'), data_dir)

    loader = deepchem.data.SDFLoader(
        tasks=qm9_tasks,
        smiles_field="smiles",
        mol_field="mol",
        featurizer=featurizer)

    dataset = loader.featurize(dataset_file)

    if split == None:
        raise ValueError()

    splitters = {
        'index': deepchem.splits.IndexSplitter(),
        'random': deepchem.splits.RandomSplitter(),
        'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=11)
    }

    splitter = splitters[split]
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)

    '''
    transformers = [
        deepchem.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=move_mean)
    ]'''
    transformers = []

    for transformer in transformers:
        train_dataset = transformer.transform(train_dataset)
        valid_dataset = transformer.transform(valid_dataset)
        test_dataset = transformer.transform(test_dataset)

    if reload:
        deepchem.utils.save.save_dataset_to_disk(
            save_folder, train_dataset, valid_dataset, test_dataset, transformers)

    return qm9_tasks, (train_dataset, valid_dataset, test_dataset), transformers


