#%%| Step 1: Import Necessary Libraries
import os
import subprocess
from prody import parsePDB, writePDB, calcCenter
#%%
# Step 2: Define Variables
pdb_id = "5wyq"  # Replace with the desired PDB ID
box_size = "30.0 30.0 30.0"
receptor_selection = "chain A and not hetero and not water"
ligand_selection = "chain A and resname SAM"

bin_path = '/home/students/bioinf/k/kl467102/miniconda3/envs/vina/bin'
reduce2_path = "/home/students/bioinf/k/kl467102/miniconda3/envs/vina/lib/python3.10/site-packages/mmtbx/command_line/reduce2.py"
mk_prepare_receptor_script = os.path.join(bin_path, 'mk_prepare_receptor.py')
geostd_path = '/home/students/bioinf/k/kl467102/vinavina/geostd/'
#%%
# Step 3: Download the PDB File
pdb_file = f"{pdb_id}.pdb"
subprocess.run(f"curl \"http://files.rcsb.org/view/{pdb_id}.pdb\" -o \"{pdb_file}\"", shell=True, check=True)
#%%
# Step 4: Parse the PDB File and Select Atoms
atoms_from_pdb = parsePDB(pdb_file)
receptor_atoms = atoms_from_pdb.select(receptor_selection)
ligand_atoms = atoms_from_pdb.select(ligand_selection)
if receptor_atoms is None:
    raise ValueError("Receptor selection did not match any atoms in the PDB file.")
if ligand_atoms is None:
    raise ValueError("Ligand selection did not match any atoms in the PDB file.")
#%%
# Step 5: Calculate Center of the Ligand
center_x, center_y, center_z = calcCenter(ligand_atoms)
print(f"Ligand center: {center_x}, {center_y}, {center_z}")
#%%
# Step 6: Save Selected Receptor and Ligand Atoms
receptor_pdb = f"{pdb_id}_receptor_atoms.pdb"
writePDB(receptor_pdb, receptor_atoms)
ligand_pdb = f"{pdb_id}_ligand.pdb"
writePDB(ligand_pdb, ligand_atoms)
#%%
# Step 7: Combine with CRYST1 Information
reduce_input_pdb = f"{pdb_id}_receptor.pdb"
os.system(f"bash -c 'cat <(grep \"CRYST1\" \"{pdb_file}\") {receptor_pdb} > {reduce_input_pdb}'")
#%%
# Step 8: Run Reduce2
reduce_opts = "approach=add add_flip_movers=True"
subprocess.run(
    f"export MMTBX_CCP4_MONOMER_LIB='{geostd_path}' && echo $MMTBX_CCP4_MONOMER_LIB && python {reduce2_path} {reduce_input_pdb} {reduce_opts}",
    shell=True, check=True, capture_output=True
)
#%%
subprocess.run(
    f"python {reduce2_path} {reduce_input_pdb} {reduce_opts}",
    shell=True, check=True
)
#%%
# Step 9: Prepare Receptor PDBQT
prepare_output = f"{pdb_id}_receptor"
subprocess.run(
    f"python {mk_prepare_receptor_script} -i {reduce_input_pdb} -o {prepare_output} -p -v "
    f"--box_center {center_x} {center_y} {center_z} --box_size {box_size} --default_altloc A",
    shell=True, check=True
)

print(f"Receptor preparation complete. Output file: {prepare_output}.pdbqt")
