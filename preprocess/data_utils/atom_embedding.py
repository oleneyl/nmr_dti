"""
From https://github.com/seokhokang/nmr_mpnn

"""
import numpy as np
from rdkit import Chem
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import string

def to_onehot(val, cat):
    vec = np.zeros(len(cat))
    for i, c in enumerate(cat):
        if val == c: vec[i] = 1

    if np.sum(vec) == 0: print('* exception: missing category', val)
    assert np.sum(vec) == 1

    return vec


def atomFeatures(aid, mol, rings, atom_list, donor_list, acceptor_list):
    def _rings(aid, rings):

        vec = np.zeros(6)
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                vec[len(ring) - 3] += 1

        return vec

    def _da(aid, donor_list, acceptor_list):

        vec = np.zeros(2)
        if aid in donor_list:
            vec[0] = 1
        elif aid in acceptor_list:
            vec[1] = 1

        return vec

    def _chiral(a):
        try:
            vec = to_onehot(a.GetProp('_CIPCode'), ['R', 'S'])
        except:
            vec = np.zeros(2)

        return vec

    a = mol.GetAtomWithIdx(aid)

    v1 = to_onehot(a.GetSymbol(), atom_list)
    v2 = to_onehot(str(a.GetHybridization()), ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])[2:]
    v3 = [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(), a.GetImplicitValence(),
          a.GetNumRadicalElectrons(), int(a.GetIsAromatic())]
    v4 = _rings(aid, rings)
    v5 = _da(aid, donor_list, acceptor_list)
    v6 = _chiral(a)

    return np.concatenate([v1, v2, v3, v4, v5, v6], axis=0)


def parse_into_embedding(mol, dim_node):
    n_max = 64

    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    atom_list = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)

    n_atom = mol.GetNumAtoms()

    rings = mol.GetRingInfo().AtomRings()

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    donor_list = []
    acceptor_list = []
    for j in range(len(feats)):
        if feats[j].GetFamily() == 'Donor':
            assert len(feats[j].GetAtomIds()) == 1
            donor_list.append(feats[j].GetAtomIds()[0])
        elif feats[j].GetFamily() == 'Acceptor':
            assert len(feats[j].GetAtomIds()) == 1
            acceptor_list.append(feats[j].GetAtomIds()[0])

    # node DV
    node = np.zeros((n_max, dim_node), dtype=np.int8)
    for j in range(n_atom):
        node[j, :] = atomFeatures(j, mol, rings, atom_list, donor_list, acceptor_list)

    return node

def create_additive_embedding(smiles, simplified_smiles, target_atom='C'):
    dim_node = 31
    mol = Chem.MolFromSmiles(smiles)
    node_map = parse_into_embedding(mol, dim_node=dim_node)

    atom_count = 0

    additive_embedding = []
    for idx, ch in enumerate(simplified_smiles):
        if ch in (string.ascii_uppercase + string.ascii_lowercase) and ch is not 'H':
            additive_embedding.append(node_map[atom_count].tolist())
            atom_count += 1
        else:
            additive_embedding.append(np.zeros([dim_node], dtype=np.int8).tolist())

    return additive_embedding
