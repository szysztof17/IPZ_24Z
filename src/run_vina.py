# coding: utf-8

import argparse
import subprocess
import os
from prody import parsePDB, writePDB, calcCenter
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import copy
from datetime import datetime
from pathlib import Path
from run_vina_series import read_box_params_from_file, prepare_ligand, dock
'''
def read_box_params_from_file(receptor_info_file):
    """
    Reads box center and size from a receptor information file.

    Args:
        receptor_info_file (str): Path to the file containing receptor box info.
    Returns:
        tuple: (box_center, box_size)
    """
    box_center = []
    box_size = []
    with open(receptor_info_file, "r") as f:
        for line in f:
            if line.startswith("center_x"):
                box_center.append(line.split("=")[1].strip())
            elif line.startswith("center_y"):
                box_center.append(line.split("=")[1].strip())
            elif line.startswith("center_z"):
                box_center.append(line.split("=")[1].strip())
            elif line.startswith("size_x"):
                box_size.append(line.split("=")[1].strip())
            elif line.startswith("size_y"):
                box_size.append(line.split("=")[1].strip())
            elif line.startswith("size_z"):
                box_size.append(line.split("=")[1].strip())

    if not box_center or not box_size:
        raise ValueError("Failed to read box center or size from receptor info file.")

    return " ".join(box_center), " ".join(box_size)


def prepare_ligand(smiles, ligand_output_name, scrub_script, mk_prepare_ligand_script, pH=7, skip_tautomer=True, skip_acidbase=False):
    
    args = ""
    if skip_tautomer:
        args += "--skip_tautomer "
    if skip_acidbase:
        args += "--skip_acidbase "
    
    ligand_name = ligand_output_name.replace(".pdbqt", "")
    ligand_sdf = f"{ligand_name}_scrubbed.sdf"
    
    # Run scrub.py
    subprocess.run(
        f"python {scrub_script} \"{smiles}\" -o {ligand_sdf} --ph {pH} {args}",
        shell=True, check=True
    )
    
    # Prepare ligand PDBQT
    subprocess.run(
        f"python {mk_prepare_ligand_script} -i {ligand_sdf} -o {ligand_output_name}",
        shell=True, check=True
    )
    return ligand_sdf

def dock(receptor_pdbqt, ligand_pdbqt, config_file, output_pdbqt):

    ligand_name_pure = Path(ligand_pdbqt).stem

    ligand_name = ligand_pdbqt.replace(".pdbqt", "")

    subprocess.run(
        f"vina --receptor {receptor_pdbqt} --ligand {ligand_pdbqt} --config {config_file} --out {output_pdbqt} --exhaustiveness=24 > logs/{ligand_name_pure}_docking.log",
        shell=True, check=True
    )

'''
def main():
    parser = argparse.ArgumentParser(description="Automated docking script using SMILES input.")
    parser.add_argument("--smiles", required=True, help="SMILES string for the ligand.")
    parser.add_argument("--receptor", required=True, help="PDBQT file for the receptor (already prepared).")
    parser.add_argument("--output", default="docking_output.pdbqt", help="Output file name for docking results.")
    parser.add_argument("--receptor_info",
                        help="Path to the receptor information file containing box parameters. Default is <receptor>.box.txt.")
    args = parser.parse_args()

    # Set default receptor_info if not provided
    if not args.receptor_info:
        args.receptor_info = f"{os.path.splitext(args.receptor)[0]}.box.txt"


    smiles = args.smiles
    receptor_pdbqt = args.receptor
    output_file = args.output
    receptor_info_file = args.receptor_info

    bin_path = '/home/students/bioinf/k/kl467102/miniconda3/envs/vina/bin'

    # Commandline scripts
    scrub_script = os.path.join(bin_path, 'scrub.py')
    mk_prepare_ligand_script = os.path.join(bin_path, 'mk_prepare_ligand.py')
    mk_export_script = os.path.join(bin_path, 'mk_export.py')


    # Create ligand output name based on the current date and hour
    current_time = datetime.now().strftime("%d_%m_%H%M")
    ligand_pdbqt = f"ligand_{current_time}.pdbqt"

    ligand_sdf = prepare_ligand(smiles, ligand_pdbqt, scrub_script, mk_prepare_ligand_script)

    # Read box center and size from the receptor info file
    box_center, box_size = read_box_params_from_file(receptor_info_file)

    # Docking configuration
    config_file = "docking_config.txt"
    with open(config_file, "w") as f:
        f.write(f"center_x = {box_center.split()[0]}\n")
        f.write(f"center_y = {box_center.split()[1]}\n")
        f.write(f"center_z = {box_center.split()[2]}\n")
        f.write(f"size_x = {box_size.split()[0]}\n")
        f.write(f"size_y = {box_size.split()[1]}\n")
        f.write(f"size_z = {box_size.split()[2]}\n")

    # Docking
    dock(receptor_pdbqt, ligand_pdbqt, config_file, output_file)
    out_sdf = output_file.replace('.pdbqt','_out.sdf')
    command = f"python {mk_export_script} {output_file} -s {out_sdf}"
    subprocess.run(command, shell = True)

    print(f"Docking completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
