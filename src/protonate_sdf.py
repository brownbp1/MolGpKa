#!/usr/bin/env python

"""
protonate_sdf.py

This script reads an input .sdf or .sdf.gz file, performs pKa prediction, and protonates atoms in 3D space based on the starting 3D structure in the .sdf/.sdf.gz file. The protonated molecules are then written to an output .sdf file.

Usage:
  protonate_sdf.py -i INPUT_SDF -o OUTPUT_SDF -p PH -t TPH

Arguments:
  -i, --input_sdf   Input .sdf or .sdf.gz file
  -o, --output_sdf  Output .sdf file
  -p, --ph          Target pH value
  -t, --tph         pH tolerance value

Example:
  protonate_sdf.py -i input.sdf -o output.sdf -p 7.0 -t 2.5
"""

import argparse
from predict_pka import predict_for_protonate
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier
from rdkit import RDLogger

from itertools import combinations

def read_sdf_file(file_path):
    """
    Read an .sdf or .sdf.gz file and return a list of RDKit molecule objects.

    Parameters:
    - file_path (str): Path to the input .sdf or .sdf.gz file.

    Returns:
    - list: List of RDKit molecule objects.
    """
    supplier = SDMolSupplier(file_path)
    mol_list = [mol for mol in supplier if mol is not None]
    return mol_list

def write_sdf_file(mol_list, output_path):
    """
    Write a list of RDKit molecule objects to an .sdf file.

    Parameters:
    - mol_list (list): List of RDKit molecule objects.
    - output_path (str): Path for the output .sdf file.
    """
    writer = Chem.SDWriter(output_path)
    for mol in mol_list:
        writer.write(mol)
    writer.close()

def modify_mol(mol, acid_dict, base_dict):
    """
    Modify RDKit molecule based on acid and base dictionaries.

    Parameters:
    - mol (RDKit Mol): RDKit molecule object.
    - acid_dict (dict): Dictionary of acid indices and corresponding pKa values.
    - base_dict (dict): Dictionary of base indices and corresponding pKa values.

    Returns:
    - RDKit Mol: Modified RDKit molecule object.
    """
    for at in mol.GetAtoms():
        idx = at.GetIdx()
        if idx in set(acid_dict.keys()):
            value = acid_dict[idx]
            nat = at.GetNeighbors()[0]
            nat.SetProp("ionization", "A")
            nat.SetProp("pKa", str(value))
        elif idx in set(base_dict.keys()):
            value = base_dict[idx]
            at.SetProp("ionization", "B")
            at.SetProp("pKa", str(value))
        else:
            at.SetProp("ionization", "O")
    nmol = AllChem.RemoveHs(mol)
    return nmol

def get_pKa_data(mol, ph, tph):
    """
    Get stable and unstable pKa data based on RDKit molecule and pH values.

    Parameters:
    - mol (RDKit Mol): RDKit molecule object.
    - ph (float): Target pH value.
    - tph (float): pH tolerance value.

    Returns:
    - tuple: Tuple containing lists of stable and unstable pKa data.
    """
    stable_data, unstable_data = [], []
    for at in mol.GetAtoms():
        props = at.GetPropsAsDict()
        acid_or_basic = props.get('ionization', False)
        pKa = float(props.get('pKa', False))
        idx = at.GetIdx()
        if acid_or_basic == "A":
            if pKa < ph - tph:
                stable_data.append([idx, pKa, "A"])
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append([idx, pKa, "A"])
        elif acid_or_basic == "B":
            if pKa > ph + tph:
                stable_data.append([idx, pKa, "B"])
            elif ph - tph <= pKa <= ph + tph:
                unstable_data.append([idx, pKa, "B"])
        else:
            continue
    return stable_data, unstable_data

def modify_acid(at):
    """
    Modify RDKit atom for acidic protonation.

    Parameters:
    - at (RDKit Atom): RDKit atom object.
    """
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(-1)
    at.SetNumExplicitHs(hnum - 1)
    return

def modify_base(at):
    """
    Modify RDKit atom for basic protonation.

    Parameters:
    - at (RDKit Atom): RDKit atom object.
    """
    hnum = at.GetNumExplicitHs()
    at.SetFormalCharge(1)
    at.SetNumExplicitHs(hnum + 1)
    return

def modify_stable_pka(new_mol, stable_data):
    """
    Modify RDKit molecule for stable pKa values.

    Parameters:
    - new_mol (RDKit Mol): RDKit molecule object.
    - stable_data (list): List of stable pKa data.
    """
    for pka_data in stable_data:
        idx, pka, acid_or_basic = pka_data
        at = new_mol.GetAtomWithIdx(idx)
        if acid_or_basic == "A":
            modify_acid(at)
        elif acid_or_basic == "B":
            modify_base(at)
    return

def modify_unstable_pka(mol, unstable_data, i):
    """
    Generate modified RDKit molecules for unstable pKa values.

    Parameters:
    - mol (RDKit Mol): RDKit molecule object.
    - unstable_data (list): List of unstable pKa data.
    - i (int): Number of atoms to modify simultaneously.

    Returns:
    - list: List of modified RDKit molecules.
    """
    combine_pka_datas = list(combinations(unstable_data, i))
    new_unmols = []
    for pka_datas in combine_pka_datas:
        new_mol = deepcopy(mol)
        if len(pka_datas) == 0:
            continue
        for pka_data in pka_datas:
            idx, pka, acid_or_basic = pka_data
            at = new_mol.GetAtomWithIdx(idx)
            if acid_or_basic == "A":
                modify_acid(at)
            elif acid_or_basic == "B":
                modify_base(at)
        new_unmols.append(new_mol)
    return new_unmols

def protonate_mol(mol, ph, tph):
    """
    Protonate an RDKit molecule based on pKa prediction.

    Parameters:
    - mol (RDKit Mol): RDKit molecule object.
    - ph (float): Target pH value.
    - tph (float): pH tolerance value.

    Returns:
    - list: List of protonated RDKit molecule objects.
    """
    obase_dict, oacid_dict, omol = predict_for_protonate(mol)
    mc = modify_mol(omol, oacid_dict, obase_dict)
    stable_data, unstable_data = get_pKa_data(mc, ph, tph)
    new_mols = []
    n = len(unstable_data)
    if n == 0:
        new_mol = deepcopy(mc)
        modify_stable_pka(new_mol, stable_data)
        new_mols.append(new_mol)
    else:
        for i in range(n + 1):
            new_mol = deepcopy(mc)
            modify_stable_pka(new_mol, stable_data)
            new_unmols = modify_unstable_pka(new_mol, unstable_data, i)
            new_mols.extend(new_unmols)
    return new_mols

def argument_parser():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Perform pKa-based protonation on molecules in an .sdf file.")
    parser.add_argument("-i", "--input_sdf", required=True, help="Input .sdf or .sdf.gz file")
    parser.add_argument("-o", "--output_sdf", required=True, help="Output .sdf file")
    parser.add_argument("-p", "--ph", type=float, required=True, help="Target pH value")
    parser.add_argument("-t", "--tph", type=float, required=True, help="pH tolerance value")
    return parser.parse_args()

def main():
    """
    Main function to execute the protonation process based on command line arguments.
    """
    args = argument_parser()

    input_sdf_path = args.input_sdf
    output_sdf_path = args.output_sdf
    ph_value = args.ph
    tph_value = args.tph

    mol_list = read_sdf_file(input_sdf_path)
    protonated_mol_list = []

    for mol in mol_list:
        pt_mols = protonate_mol(mol, ph=ph_value, tph=tph_value)
        protonated_mol_list.extend(pt_mols)

    write_sdf_file(protonated_mol_list, output_sdf_path)


if __name__ == "__main__":
    main()

