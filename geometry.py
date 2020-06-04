from rdkit import Chem
from rdkit.Chem import AllChem
# Aspirine
TEST_SMILES = 'CC(=O)OC1=CC=CC=C1C(=O)O'


mol = Chem.MolFromSmiles(TEST_SMILES)
result = AllChem.EmbedMolecule(mol)
dis_mat = AllChem.Get3DDistanceMatrix(mol)
pos = mol.GetConformer().GetAtomPosition(0)
print(mol)
print(result)
print(dis_mat)
print('++++')
print(pos)
