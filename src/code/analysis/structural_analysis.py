import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import (MolToSmiles, MolFromSmiles,
                        RemoveAllHs)
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from rdkit import DataStructs
from tqdm import tqdm
import os
import os.path as osp

import pandas as pd
import argparse


from ..dataset.xyz2mol_final import XYZ2MOL


import logging



AVAILABLE_BO = {
    'P': [3, 5],
    'S': [2, 4, 6],
    'As': [3, 5],
    'Se': [2, 4, 6],
}

def canon_ignore_metal_explicit_H(mol, metals=("Al",), sanitize=True):
    if mol is None:
        return None

    rw = Chem.RWMol(mol)  # 可写副本
    metals = set(metals)

    for a in rw.GetAtoms():
        if a.GetSymbol() in metals:
            a.SetNumExplicitHs(0)   # 去掉 [AlH2] 这种显式H计数
            a.SetNoImplicit(True)   # 防止 RDKit 又补回隐式H

    out = rw.GetMol()
    if sanitize:
        try:
            Chem.SanitizeMol(out)
        except Exception:
            # 金属体系 sanitize 失败很常见：宁可继续也别直接崩
            pass
    return out

def get_clean_smi(mol, ignore_metal_explicit_H=False, metals=("Al",)):
    if mol is None:
        return None

    mol = RemoveAllHs(mol)

    if ignore_metal_explicit_H:
        mol = canon_ignore_metal_explicit_H(mol, metals=metals, sanitize=True)
        if mol is None:
            return None

    Chem.RemoveStereochemistry(mol)

    return MolToSmiles(mol, canonical=True)

def check_quiv_by_inchikey(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2) 
    return Chem.MolToInchiKey(mol1)==Chem.MolToInchiKey(mol2)

def tanimoto_similarity(pred_smi, tgt_smi):
    pred_mol, tgt_mol = MolFromSmiles(pred_smi), MolFromSmiles(tgt_smi)
    fpgen = AllChem.GetRDKitFPGenerator()
    pred_fp, tgt_fp = fpgen.GetFingerprint(pred_mol), fpgen.GetFingerprint(tgt_mol)
    return DataStructs.TanimotoSimilarity(pred_fp, tgt_fp)

def tanimoto_similarity_morgan(pred_smi, tgt_smi):
    pred_mol, tgt_mol = MolFromSmiles(pred_smi), MolFromSmiles(tgt_smi)
    fpgen = GetMorganGenerator(radius=1, fpSize=2048)
    pred_fp, tgt_fp = fpgen.GetFingerprint(pred_mol), fpgen.GetFingerprint(tgt_mol)
    return DataStructs.TanimotoSimilarity(pred_fp, tgt_fp)


def cal_mean_max_similarity(all_tanimoto_similarity_list):
    array = np.array(all_tanimoto_similarity_list, dtype=float)
    average = np.nanmean(array)
    max_similarity = np.nanmax(array, axis=1)
    average_max = np.nanmean(max_similarity)

    return average, average_max
   


def bond_order_sum(atom: Chem.Atom) -> float:
    s = 0.0
    for b in atom.GetBonds():
        if b.GetIsAromatic():
            s += 1.5
        else:
            s += float(b.GetBondTypeAsDouble())
    return s

def check_radical_electrons(mol):
    is_radical = False
   
    for atom in mol.GetAtoms():
        num_radical_electrons = atom.GetNumRadicalElectrons()
        if num_radical_electrons > 0:
            if atom.GetSymbol() in AVAILABLE_BO.keys():
                bo = bond_order_sum(atom)
                # print(f'bo: {bo}')
                # print(f'As | num_radical_electrons:{num_radical_electrons}, bond order: {bo}')
                if bo in AVAILABLE_BO[atom.GetSymbol()]:
                    continue
                else:
                    is_radical = True
            elif atom.GetSymbol() == "Al":
                continue
            else:
                is_radical = True

    return is_radical

def check_connectivity(mol):
    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
    return len(mol_frags) == 1, mol_frags

def main(data_file, smi_origin, log_file_name=None, save_path=None, 
         debug=False, debug_data=10, dataset="qm9s"):
    """
    smi_origin: 
        list, from qm9s csv; None
    """
    if log_file_name is None: log_file_name='sample_analysis.log'
    elif '.log' not in log_file_name: log_file_name += '.log'

    log_file_name = os.path.join(save_path, log_file_name)
    print(f"Log file: {log_file_name}")
    logging.basicConfig(filename=log_file_name,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing sample results: {data_file}")

    data = np.load(data_file)
    pos_pred = data["pos_pred"] # (num_mol, num_sample, max_num_nodes, dims)
    pos_tgt = data["pos_tgt"] # (num_mol, max_num_nodes, dims)
    charges = data["charges"]
    node_mask = data["node_mask"]
    if debug:
        pos_pred = pos_pred[:debug_data]
        pos_tgt = pos_tgt[:debug_data]
        charges = charges[:debug_data]
        node_mask = node_mask[:debug_data]

    num_atoms_all = np.sum(node_mask,axis=1,dtype=int)
    num_mol, num_sample, max_num_nodes, dims = pos_pred.shape


    # process tgt mol
    ignore_idx_list = []
    for i in tqdm(range(num_mol)):
        if i in ignore_idx_list: continue

        num_atoms_i = num_atoms_all[i]
        charges_i = np.squeeze(charges[i].astype(np.int16))
        if dataset== "qm9s":
            mol = XYZ2MOL(atom_charge=charges_i, pos=pos_tgt[i], num_atoms=num_atoms_i, output_log=False, bond_threshold=0.4)
            mol, smi_not_canonical = mol.xyz2mol(process_fragments=False)
        elif dataset == "qme14s":
            MOL = XYZ2MOL(atom_charge=charges_i, pos=pos_tgt[i], num_atoms=num_atoms_i, output_log=False, bond_threshold=0.3)
            mol, smi_not_canonical = MOL.xyz2mol(process_fragments=False)

        try:
            smi_tgt = get_clean_smi(mol=mol)
        except Exception as e:
            print("smi_not_canonical", smi_not_canonical)
            print("smi_origin", smi_origin[i])
            print("check_quiv_by_inchikey", check_quiv_by_inchikey(smi1=smi_not_canonical, smi2=smi_origin[i]))
            if not check_quiv_by_inchikey(smi1=smi_not_canonical, smi2=smi_origin[i]):
                print(f"Ignore mol {i}, (smi from dataset {smi_origin[i]}) : {e}")
                ignore_idx_list.append(i)
        else:
            if "Al" in smi_origin[i]:
                smi_origin_i = get_clean_smi(MolFromSmiles(smi_origin[i]), ignore_metal_explicit_H=True)
                smi_tgt = get_clean_smi(mol=mol, ignore_metal_explicit_H=True)
            else:
                smi_origin_i = get_clean_smi(MolFromSmiles(smi_origin[i]))
            
            if not check_quiv_by_inchikey(smi1=smi_tgt, smi2=smi_origin_i):
                print("smi_not_canonical", smi_not_canonical)
                raise ValueError(f"{i} | Converted smiles {smi_tgt} and smiles from {dataset} {smi_origin_i} do not match.")


    # process sampled mol
    logger.info("Finish processing tgt mol. Start processing sampled mol...")
    valid_sampled_mol = 0
    num_not_connected_mol = 0
    num_unstable_mol = 0
    num_mol_smi_match = 0
    all_similarity_list = []
    all_stable_similarity_list = []
    all_sample_results = {'mol_idx': [], 'smi_tgt': [],
                          'sample_idx':[], 'smi_sampled': [], 
                          'similarity': [], 'stable_similarity': [],
                          'is_smi_matched': []}
    for i in tqdm(range(num_mol)):
        if i in ignore_idx_list: continue

        num_atoms_i = num_atoms_all[i]
        charges_i = charges[i]
        if "Al" in smi_origin[i]:
            smi_tgt = get_clean_smi(MolFromSmiles(smi_origin[i]), ignore_metal_explicit_H=True)
        else:
            smi_tgt = get_clean_smi(MolFromSmiles(smi_origin[i]))
   
        
            
        logger.info(f"mol {i} | ref mol: {smi_tgt}")
     
        pos_pred_i = pos_pred[i]
        smi_pred_list = [None] * num_sample
        similarity_list = [None] * num_sample
        stable_similarity_list = [None] * num_sample
        # stable_similarity_scaffold_list = [None] * num_sample
        is_smi_match_list = [0] * num_sample
        # num_smi_match = 0
        num_valid_sampled = 0
        for j in range(num_sample):
            if dataset== "qm9s":
                mol = XYZ2MOL(atom_charge=charges_i, pos=pos_pred_i[j], num_atoms=num_atoms_i, output_log=False, bond_threshold=0.4)
                mol_ij, smi_not_canonical = mol.xyz2mol(process_fragments=True)
                
            elif dataset == "qme14s":
                mol = XYZ2MOL(atom_charge=charges_i, pos=pos_pred_i[j], num_atoms=num_atoms_i, output_log=False, bond_threshold=0.3)
                mol_ij, smi_not_canonical = mol.xyz2mol(process_fragments=True)

            try:
                smi_ij = get_clean_smi(mol=mol_ij)
            except Exception as e:
                logger.error(f"mol {i} | sampled mol {j} | smiles from geo: {smi_not_canonical} | {e}")
                continue
            
            try:
                is_connected, mol_frags = check_connectivity(mol_ij)
            except Exception as e:
                logger.error(f"mol {i} | sampled mol {j} | smiles from geo: {smi_not_canonical} | {e}")
                continue

            num_valid_sampled += 1
            if not is_connected:
                num_not_connected_mol += 1
                mol_ij = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms()) # get the largest fragment
                logger.warning(f"mol {i} | sampled mol {j}: {smi_ij} is not connected. max fragment: {get_clean_smi(mol=mol_ij)}")
                smi_ij = get_clean_smi(mol=mol_ij)
            smi_pred_list[j] = smi_ij
            
            stable = True
            # check if the sampled mol is stable
            if check_radical_electrons(mol_ij):
                logger.warning(f"mol {i} | sampled mol {j}: {smi_ij} has radical electrons.")
                num_unstable_mol += 1
                stable = False
            elif Chem.GetFormalCharge(mol_ij) != 0:
                logger.warning(f"mol {i} | sampled mol {j}: {smi_ij} with formal charge {Chem.GetFormalCharge(mol_ij)}.")
                num_unstable_mol += 1
                stable = False

            # calculate graph similarity
            similarity = tanimoto_similarity_morgan(pred_smi=smi_ij, tgt_smi=smi_tgt)
            similarity_list[j] = similarity

            if stable:
                stable_similarity_list[j] = similarity
            

            if "Al" in smi_tgt:
                smi_ij = get_clean_smi(mol=mol_ij, ignore_metal_explicit_H=True)
            if check_quiv_by_inchikey(smi1=smi_ij, smi2=smi_tgt):
                is_smi_match_list[j] = 1
        
        num_sample_matched = sum(is_smi_match_list)
        is_smi_match = 0
        if num_sample_matched > 0 : is_smi_match = 1
        num_mol_smi_match += is_smi_match

        valid_sampled_mol += num_valid_sampled
        # all_smi_pred_list.append(smi_pred_list)
        all_similarity_list.append(similarity_list)
        all_stable_similarity_list.append(stable_similarity_list)
        # all_stable_similarity_scaffold_list.append(stable_similarity_scaffold_list)
        
        all_sample_results['mol_idx'] += [i]*num_sample
        all_sample_results['smi_tgt'] += [smi_tgt]*num_sample
        all_sample_results['sample_idx'] += list(range(num_sample))
        all_sample_results['smi_sampled'] += smi_pred_list
        all_sample_results['similarity'] += similarity_list
        all_sample_results['stable_similarity'] += stable_similarity_list
        # all_sample_results['stable_similarity_scaffold'] += all_stable_similarity_scaffold_list
        all_sample_results['is_smi_matched'] += is_smi_match_list
        

    for k, v in all_sample_results.items():
        print(k, len(v))
    results_df = pd.DataFrame(all_sample_results)
    results_df.to_csv(log_file_name.replace('.log', '.csv'), index=False)

    num_mol = num_mol - len(ignore_idx_list)
    num_all_sampled = num_mol * num_sample
    

    
    logger.info(f"num_test_data: {num_mol} | num_sample: {num_sample}")
    # logger.info(f"num invalid tgt mol: {invalid_tgt_mol} | vadility: 1-{invalid_tgt_mol}/{num_mol}={'%.4f' % (1-invalid_tgt_mol/num_mol)}")
    logger.info(f"num valid sampled mol: {valid_sampled_mol} | vadility: {valid_sampled_mol}/{num_all_sampled}={'%.4f' % (valid_sampled_mol/num_all_sampled)}")
    
    num_stable_mol = valid_sampled_mol - num_unstable_mol
    logger.info(f"num stable sampled mol: {valid_sampled_mol}-{num_unstable_mol}={num_stable_mol} | mol stability: {num_stable_mol}/{num_all_sampled}={'%.4f' % (num_stable_mol/num_all_sampled)}")
    
    num_connected_mol = valid_sampled_mol - num_not_connected_mol
    logger.info(f"num connected sampled mol: {valid_sampled_mol}-{num_not_connected_mol}={num_connected_mol} | connectivity: {num_connected_mol}/{num_all_sampled}={'%.4f' % (num_connected_mol/num_all_sampled)}")
    
    logger.info(f"num_smi_match_mol: {num_mol_smi_match} | smi acc: {num_mol_smi_match}/{num_mol} = {'%.4f' % (num_mol_smi_match/num_mol)}")

    similarity, max_similarity = cal_mean_max_similarity(all_similarity_list)
    logger.info(f"similarity | average: {'%.4f' % similarity} | max: {max_similarity}")
    stable_similarity, max_stable_similarity = cal_mean_max_similarity(all_stable_similarity_list)
    logger.info(f"stable similarity | average: {'%.4f' % stable_similarity} | max: {max_stable_similarity}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_file', type=str, default=None,
                        help='sampled file, npz format')
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--log_name', type=str, default="analyze.log")
    parser.add_argument('--save_path', type=str, default="./")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_data', type=int, default=10)
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], default="qm9s")
    args = parser.parse_args()
    
    
    dir = os.path.dirname(args.sample_file)
    if args.save_path == "./":
        save_path = dir
    else:
        save_path = args.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if args.test_data is not None:
        test_data = pd.read_pickle(args.test_data)
        smi_origin = test_data['smiles'].tolist()
    else:
        smi_origin = None

    main(data_file=args.sample_file, 
         smi_origin=smi_origin, 
         log_file_name=args.log_name, 
         save_path=save_path, 
         debug=args.debug,
         debug_data=args.debug_data,
         dataset=args.dataset)