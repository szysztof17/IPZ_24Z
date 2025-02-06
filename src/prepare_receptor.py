# receptor_preparation.py

import argparse
import subprocess
import os
from prody import parsePDB, writePDB, calcCenter
from openbabel import pybel


def prepare_receptor_from_pdb_id(
    pdb_id, 
    receptor_selection, 
    ligand_selection, 
    reduce2_path, 
    mk_prepare_receptor_script, 
    geostd_path, 
    box_center, 
    box_size
):
    """
    Prepares a receptor from a PDB ID.

    Args:
        pdb_id (str): PDB ID for the receptor.
        receptor_selection (str): ProDy selection for receptor atoms.
        ligand_selection (str): ProDy selection for ligand atoms.
        reduce2_path (str): Path to the reduce2.py script.
        mk_prepare_receptor_script (str): Path to the receptor preparation script.
        geostd_path (str): Path to geostd files.
        box_center (str): Box center coordinates for docking.
        box_size (str): Box size dimensions for docking.
    Returns:
        str: Path to the prepared receptor file (PDBQT).
    """

    # Download PDB file
    pdb_file = f"{pdb_id}.pdb"
    subprocess.run(f"curl \"http://files.rcsb.org/view/{pdb_id}.pdb\" -o \"{pdb_file}\"", shell=True, check=True)
    # Parse PDB and select atoms
    atoms_from_pdb = parsePDB(pdb_file)
    receptor_atoms = atoms_from_pdb.select(receptor_selection)
    ligand_atoms = atoms_from_pdb.select(ligand_selection)
    
    if receptor_atoms is None:
        raise ValueError("Receptor selection did not match any atoms in the PDB file.")
    if ligand_atoms is None:
        raise ValueError("Ligand selection did not match any atoms in the PDB file.")
    center_x, center_y, center_z = calcCenter(ligand_atoms)
    receptor_pdb = f"{pdb_id}_receptor_atoms.pdb"
    writePDB(receptor_pdb, receptor_atoms)
    ligand_pdb = f"{pdb_id}_ligand.pdb"
    writePDB(ligand_pdb, ligand_atoms)
    print(f"Ligand atoms saved to: {ligand_pdb}")
    # First, create a pybel molecule object from the ligand atoms
    ligand_mol = pybel.readfile("pdb", ligand_pdb).__next__()

    # Add explicit hydrogens
    ligand_mol.addh()

    # Write ligand to SDF
    ligand_sdf = f"{pdb_id}_ligand.sdf"
    ligand_mol.write("sdf", ligand_sdf)

    # Combine with CRYST1 information
    reduce_input_pdb = f"{pdb_id}_receptor.pdb"
    os.system(f"bash -c 'cat <(grep \"CRYST1\" \"{pdb_file}\") {receptor_pdb} > {reduce_input_pdb}'")
    
    # Run reduce2
    reduce_opts = "approach=add add_flip_movers=True"
    subprocess.run(
        f"export MMTBX_CCP4_MONOMER_LIB=\"{geostd_path}\"; python {reduce2_path} {reduce_input_pdb} {reduce_opts}",
        shell=True, check=True
    )
    
    # Prepare receptor PDBQT
    prepare_output = f"{pdb_id}_receptor"
    subprocess.run(
        f"python {mk_prepare_receptor_script} -i {reduce_input_pdb} -o {prepare_output} -p -v --box_center {center_x} {center_y} {center_z} --box_size {box_size} --default_altloc A",
        shell=True, check=True
    )
    return f"{prepare_output}.pdbqt"

def main():
    parser = argparse.ArgumentParser(description="Receptor preparation script using a PDB ID.")
    parser.add_argument("--pdb_id", required=True, help="PDB ID for the receptor.")
    parser.add_argument("--box_size", required=True, help="Box size for docking (e.g., '20.0 20.0 20.0').")
    parser.add_argument("--receptor_selection", default="chain A and not water and not hetero", help="Atom selection for receptor.")
    parser.add_argument("--ligand_selection", default="chain A and resname 9D6", help="Atom selection for ligand.")
    args = parser.parse_args()
    
    # Paths to required scripts
    bin_path = '/home/students/bioinf/k/kl467102/miniconda3/envs/vina/bin'
    reduce2_path = "/home/students/bioinf/k/kl467102/miniconda3/envs/vina/lib/python3.10/site-packages/mmtbx/command_line/reduce2.py"
    mk_prepare_receptor_script = os.path.join(bin_path, 'mk_prepare_receptor.py')
    geostd_path  = "/home/students/bioinf/k/kl467102/vinavina/geostd"

    
    # Box center will be computed from ligand in the PDB
    pdb_id = args.pdb_id
    box_size = args.box_size
    receptor_selection = args.receptor_selection
    ligand_selection = args.ligand_selection
    
    receptor_pdbqt = prepare_receptor_from_pdb_id(
        pdb_id=pdb_id, 
        receptor_selection=receptor_selection, 
        ligand_selection=ligand_selection, 
        reduce2_path=reduce2_path, 
        mk_prepare_receptor_script=mk_prepare_receptor_script, 
        geostd_path=geostd_path, 
        box_center=None,  # Center will be auto-calculated
        box_size=box_size
    )
    print(f"Receptor preparation complete. Output file: {receptor_pdbqt}")

if __name__ == "__main__":
    main()
'''
EXAMPLE RUN
!python src/prepare_receptor.py --pdb_id "2zzn" --box_size "30.0 30.0 30.0"\
     --ligand_selection "chain A and resname SAM" \
     --receptor_selection "chain A and not hetero and not water"

'''