from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "CC1=CC(=O)OC2=C1C=C3CCCN4C3=C2CCC4"  # Coumarin 102 (C102), CAS 41267-76-9
mol = Chem.MolFromSmiles(smiles)
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
with open("coumarin102.xyz", "w") as f:
    f.write(f"{mol.GetNumAtoms()}\nCoumarin 102 (C102), MMFF-optimized geometry\n")
    f.write(xyz_block + "\n")

print(f"Atomos: {mol.GetNumAtoms()}")
print(f"Formula: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
