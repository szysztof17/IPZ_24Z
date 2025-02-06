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

def prepare_ligand(smiles, ligand_output_name, scrub_script, mk_prepare_ligand_script, pH=7.35, skip_tautomer=True, skip_acidbase=False):
    
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

def dock(receptor_pdbqt, ligand_pdbqt, config_file, output_pdbqt, output_file, mk_export_script, exhaustiveness, logs):

    ligand_name_pure = Path(ligand_pdbqt).stem

    ligand_name = ligand_pdbqt.replace(".pdbqt", "")

    subprocess.run(
        f"vina --receptor {receptor_pdbqt} --ligand {ligand_pdbqt} --config {config_file} --out {output_pdbqt} --exhaustiveness={exhaustiveness} > {logs}/{ligand_name_pure}_docking.log",
        shell=True, check=True
    )

    command = f"python {mk_export_script} {output_pdbqt}  -s {ligand_name}_out.sdf"
    subprocess.run(command, shell = True)


def main():
    parser = argparse.ArgumentParser(description="Automated docking script using SMILES input.")
    parser.add_argument("--smiles_file", required=True, help="File containing SMILES strings for the ligands.")
    parser.add_argument("--receptor", required=True, help="PDBQT file for the receptor (already prepared).")
    parser.add_argument("--output_dir", default="docking_outputs", help="Directory for output docking results.")
    parser.add_argument("--receptor_info",
                        help="Path to the receptor information file containing box parameters. Default is <receptor>.box.txt.")
    parser.add_argument("--exhaustiveness", default=8, help="VINA exhaustiveness parameter. Default is 32.")

    args = parser.parse_args()

    # Set default receptor_info if not provided
    if not args.receptor_info:
        args.receptor_info = f"{os.path.splitext(args.receptor)[0]}.box.txt"

    receptor_pdbqt = args.receptor
    receptor_info_file = args.receptor_info
    output_dir = args.output_dir
    exhaustiveness = args.exhaustiveness
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Create the output directories if they do not exist
    ligands_dir = os.path.join(output_dir, "ligands")
    results_dir = os.path.join(output_dir, "results")
    logs_dir = os.path.join(output_dir,'logs')

    os.makedirs(ligands_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


    bin_path = '/home/students/bioinf/k/kl467102/miniconda3/envs/vina/bin'

    # Commandline scripts
    scrub_script = os.path.join(bin_path, 'scrub.py')
    mk_prepare_ligand_script = os.path.join(bin_path, 'mk_prepare_ligand.py')
    mk_export_script = os.path.join(bin_path, 'mk_export.py')

    # Read SMILES strings and IDs from file
    with open(args.smiles_file, "r") as f:
        # Skip the header line
        next(f)
        smiles_list = [line.strip().split() for line in f.readlines()]
        # Assuming the file has two columns: ID and SMILES

    for i, (ligand_id, smiles) in enumerate(smiles_list):
        print(f"Running docking for ligand {i + 1}/{len(smiles_list)}: {ligand_id} ({smiles})")

        ligand_pdbqt = os.path.join(ligands_dir, f"ligand_{ligand_id}.pdbqt")

        ligand_sdf = prepare_ligand(smiles, ligand_pdbqt, scrub_script, mk_prepare_ligand_script)

        # Docking
        output_file = f"docking_output_{ligand_id}.pdbqt"
        output_pdbqt = os.path.join(results_dir, f"{output_file.replace('.pdbqt', f'_{ligand_id}.pdbqt')}")
        dock(receptor_pdbqt, ligand_pdbqt, receptor_info_file, output_pdbqt, output_file, mk_export_script, exhaustiveness, logs_dir)
        print(f"Docking completed for ligand {ligand_id}. Results saved to {output_pdbqt}")


if __name__ == "__main__":
    main()

'''
EXAMPLE RUN
python src/run_vina_series.py --smiles_file bio_pre_filtered_sim.tsv\
    --receptor 5WYQ/5wyq_receptor.pdbqt\
    --output_dir 5WYQ_docking \
    --exhaustiveness 32
'''