
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import argparse


def xyz2gjf(onehot, pos, num_atoms, save_dir, filename, NProcShared, index2type):
    atoms = np.argmax(onehot[:num_atoms, :],  axis=1).tolist()
    atomic_symbols = [index2type[int(k)] for k in atoms]
    xyz_coordinates = pos[:num_atoms].tolist()

    with open(os.path.join(save_dir, f"{filename}.gjf"), "w") as f_:
        f_.write(f"%Chk={filename}.chk\n")
        f_.write(f"%Mem={int(NProcShared*2)}GB\n")
        f_.write(f"%NProcShared={NProcShared}\n")
        f_.write(f"# B3LYP/def2TZVP opt freq\n")
        f_.write("\n")
        f_.write(f"{filename}\n")
        f_.write("\n")
        f_.write("0 1\n")
        for i in range(num_atoms):
            xyz_i = xyz_coordinates[i]
            f_.write(f"{atomic_symbols[i]} {' '.join([str(j) for j in xyz_i])}\n")
        f_.write("\n")

def check_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def main_xyz2gjf(data_file, NProcShared, dataset,
                structral_analysis_result, save_dir=None,
                start_mol=None, end_mol=None,
                num_mol_compute=None, num_sample_compute=None,
                tgt_mol=False):
    """
    args:
        data_file: .npz file
        save_dir: str, default is None
    """
    if dataset == 'qm9s':
        INDEX2TYPE = {0: "H", 1: "C", 2:"N", 3:"O", 4:"F"}
    elif dataset == 'qme14s':
        INDEX2TYPE = {0: "H", 
                      1: "B",  2: "C",  3: "N", 4: "O",  5: "F",
                      6: "Al", 7: "Si", 8: "P", 9: "S",  10: "Cl",
                                       11:"As", 12:"Se", 13: "Br"}

    if type(structral_analysis_result) == str:
        structral_analysis_result = pd.read_csv(structral_analysis_result)
    data = np.load(data_file)
    pos_pred = data["pos_pred"] # (num_mol, num_sample, max_num_nodes, dims)
    pos_tgt = data["pos_tgt"] # (num_mol, max_num_nodes, dims)
    onehot = data["onehot"]
    node_mask = data["node_mask"]

    num_atoms_all = np.sum(node_mask,axis=1,dtype=int)
    num_mol, num_sample, max_num_nodes, dims = pos_pred.shape
    
    if start_mol is not None and end_mol is not None:
        num_mol_compute = end_mol - start_mol + 1
    elif num_mol_compute is None: 
        num_mol_compute = num_mol

    if num_sample_compute is None: num_sample_compute = num_sample
    assert (num_mol_compute <= num_mol) & (num_sample_compute <= num_sample)

    if save_dir is None:
        save_dir = os.path.dirname(data_file)
        dir_name = data_file.split("/")[-1].split(".")[0]
        save_dir = os.path.join(save_dir, f"gjf_{dir_name}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    idx = 0
    for i in tqdm(range(num_mol_compute)):
        if start_mol is not None:
            i = start_mol + i
        num_atoms_i = num_atoms_all[i]
        onehot_i = onehot[i]
        
        if tgt_mol:
            file_name_tgt = f"mol_{i}_tgt"
            save_dir_i_tgt = os.path.join(save_dir, f"{idx}_mol_{i}_tgt")
            idx += 1
            check_dir(save_dir_i_tgt)
            xyz2gjf(onehot=onehot_i, pos=pos_tgt[i], num_atoms=num_atoms_i, 
                    save_dir=save_dir_i_tgt, filename=file_name_tgt,
                    NProcShared=NProcShared,
                    index2type=INDEX2TYPE)
        
        struct_mol_i = structral_analysis_result[structral_analysis_result['mol_idx']==i]
        pos_pred_i = pos_pred[i]
        for j in tqdm(range(num_sample_compute)):
            file_name_pred = f"mol_{i}_sample_{j}"

            # check if the sample is stable
            struct_mol_i_j = struct_mol_i[struct_mol_i['sample_idx']==j]
            if struct_mol_i_j['stable_similarity'].isna().all(): 
                print(f"mol {i} sample {j} is not stable")
                continue

            # save_dir_i_j = os.path.join(save_dir_i, f"sample_{j}")
            save_dir_i_j = os.path.join(save_dir, f"{idx}_mol_{i}_sample_{j}")
            idx += 1
            check_dir(save_dir_i_j)
            if len(onehot_i.shape) == 2:
                xyz2gjf(onehot=onehot_i, pos=pos_pred_i[j], num_atoms=num_atoms_i, 
                        save_dir=save_dir_i_j, filename=file_name_pred,
                        NProcShared=NProcShared,
                        index2type=INDEX2TYPE)
            elif len(onehot_i.shape) == 3:
                xyz2gjf(onehot=onehot_i[j], pos=pos_pred_i[j], num_atoms=num_atoms_i, 
                        save_dir=save_dir_i_j, filename=file_name_pred,
                        NProcShared=NProcShared,
                        index2type=INDEX2TYPE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--dataset", type=str, default='qm9s')
    parser.add_argument("--structral_analysis_result", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--NProcShared", type=int)
    parser.add_argument("--start_mol", type=int)
    parser.add_argument("--end_mol", type=int)
    parser.add_argument("--num_mol_compute", type=int)
    parser.add_argument("--num_sample_compute", type=int)
    parser.add_argument("--tgt_mol", type=bool, help="whether to generate input for the tgt mol")
    args = parser.parse_args()
    main_xyz2gjf(data_file=args.data_file, NProcShared=args.NProcShared,
                dataset=args.dataset,
                structral_analysis_result=args.structral_analysis_result, 
                save_dir=args.save_dir,
                start_mol=args.start_mol, end_mol=args.end_mol,
                num_mol_compute=args.num_mol_compute, num_sample_compute=args.num_sample_compute,
                tgt_mol=args.tgt_mol)
