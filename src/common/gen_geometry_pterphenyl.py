"""
Geometry generation for p-terphenyl -- same procedure as gen_geometry.py (RDKit
ETKDGv3 embedding + MMFF optimization), parametrized through
molecule_config.MOLECULES["p_terphenyl"] instead of a second hardcoded SMILES
string. Does not touch gen_geometry.py or coumarin102.xyz.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from molecule_config import MOLECULES

cfg = MOLECULES["p_terphenyl"]
mol = Chem.MolFromSmiles(cfg["smiles"])
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42
cid = AllChem.EmbedMolecule(mol, params)
AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

conf = mol.GetConformer()
lines = []
for atom in mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    lines.append(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")

xyz_block = "\n".join(lines)
with open("p_terphenyl.xyz", "w") as f:
    f.write(f"{mol.GetNumAtoms()}\n{cfg['name']}, MMFF-optimized geometry\n")
    f.write(xyz_block + "\n")

print(f"Atomos: {mol.GetNumAtoms()}")
print(f"Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
