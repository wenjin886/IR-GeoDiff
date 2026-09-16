import numpy as np
import pandas as pd
from argparse import ArgumentParser
from rdkit import Chem
from tqdm import tqdm
import os
import math
import re

# import matplotlib.pyplot as plt
# import seaborn as sns
from .structural_analysis import get_clean_smi, check_quiv_by_inchikey
# import shutil
from scipy import interpolate
import logging
from ..dataset.xyz2mol_final import XYZ2MOL



INDEX2TYPE = {0: "H", 1: "C", 2:"N", 3:"O", 4:"F"}

SCALING_FACTOR = 0.965  


# def xyz2gjf(onehot, pos, num_atoms, save_dir, filename, NProcShared):
    
#     atoms = np.argmax(onehot[:num_atoms, :],  axis=1).tolist()
#     atomic_symbols = [INDEX2TYPE[int(k)] for k in atoms]
#     xyz_coordinates = pos[:num_atoms].tolist()

#     with open(os.path.join(save_dir, f"{filename}.gjf"), "w") as f_:
#         f_.write(f"%Chk={filename}.chk\n")
#         f_.write(f"%Mem={int(NProcShared*2)}GB\n")
#         f_.write(f"%NProcShared={NProcShared}\n")
#         f_.write(f"# B3LYP/def2TZVP opt freq\n")
#         f_.write("\n")
#         f_.write(f"{filename}\n")
#         f_.write("\n")
#         f_.write("0 1\n")
#         for i in range(num_atoms):
#             xyz_i = xyz_coordinates[i]
#             f_.write(f"{atomic_symbols[i]} {' '.join([str(j) for j in xyz_i])}\n")
#         f_.write("\n")

# def check_dir(dir_name):
#     if not os.path.exists(dir_name):
#         os.makedirs(dir_name)
#     return dir_name

# def main_xyz2gjf(data_file, NProcShared, num_mol_compute=None, num_sample_compute=None):
#     """
#     args:
#         data_file: .npz file
#     """
#     data = np.load(data_file)
#     pos_pred = data["pos_pred"] # (num_mol, num_sample, max_num_nodes, dims)
#     pos_tgt = data["pos_tgt"] # (num_mol, max_num_nodes, dims)
#     onehot = data["onehot"]
#     node_mask = data["node_mask"]

#     num_atoms_all = np.sum(node_mask,axis=1,dtype=int)
#     num_mol, num_sample, max_num_nodes, dims = pos_pred.shape
#     if num_mol_compute is None: num_mol_compute = num_mol
#     if num_sample_compute is None: num_sample_compute = num_sample
#     assert (num_mol_compute <= num_mol) & (num_sample_compute <= num_sample)

#     save_dir = os.path.dirname(data_file)
#     dir_name = data_file.split("/")[-1].split(".")[0]
#     save_dir = os.path.join(save_dir, f"gjf_{dir_name}")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     for i in tqdm(range(num_mol_compute)):
#         num_atoms_i = num_atoms_all[i]
#         onehot_i = onehot[i]

#         file_name_tgt = f"mol_{i}_tgt"
#         save_dir_i = os.path.join(save_dir, f"mol_{i}")
#         check_dir(save_dir_i)
#         save_dir_i_tgt = os.path.join(save_dir_i, "tgt")
#         check_dir(save_dir_i_tgt)
#         xyz2gjf(onehot=onehot_i, pos=pos_tgt[i], num_atoms=num_atoms_i, 
#                 save_dir=save_dir_i_tgt, filename=file_name_tgt,
#                 NProcShared=NProcShared)
        
#         pos_pred_i = pos_pred[i]
#         for j in tqdm(range(num_sample_compute)):
#             file_name_pred = f"mol_{i}_sample_{j}"
#             save_dir_i_j = os.path.join(save_dir_i, f"sample_{j}")
#             check_dir(save_dir_i_j)
#             xyz2gjf(onehot=onehot_i, pos=pos_pred_i[j], num_atoms=num_atoms_i, 
#                     save_dir=save_dir_i_j, filename=file_name_pred,
#                     NProcShared=NProcShared)

# def norm_spectrum(
#     spectrum: np.ndarray, bounds=(0, 99)
# ) -> np.ndarray:
#     spectrum_norm = spectrum / max(spectrum) * bounds[1]
#     spectrum_norm_int = spectrum_norm.astype(int)
#     spectrum_norm_int = np.clip(spectrum_norm_int, *bounds)
#     spectrum_norm_int = spectrum_norm_int/max(spectrum_norm_int)
#     return spectrum_norm_int

def parse_gaussian_log(log_file, scale=True):
    frequencies = []
    intensities = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Frequencies --" in line:
                freqs = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                frequencies.extend(freqs)  
            elif "IR Inten    --" in line:
                ints = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                intensities.extend(ints)  

    if scale:
        frequencies = np.array(frequencies) * SCALING_FACTOR
        intensities = np.array(intensities)

    return frequencies, intensities

def lorentzian(x, x0, F):
    """
    Lorentzian function to broaden spec
    args:
        x0: wavenumber of one vibrational mode
        F: half-width of the peak
    """
    return (F / (2 *np.pi)) / ((x - x0)**2 + 0.25 * F**2) 

def broaden_spectrum(frequencies, intensities, F=15, 
                     x_values=np.linspace(500, 4000, 3501),
                     ):
    """
    args:
        F: half-width of the peak
    """
    spectrum = np.zeros_like(x_values)
    
    for f, i in zip(frequencies, intensities):
        spectrum += i * lorentzian(x_values, f, F)  # 每个峰展宽
    
    return spectrum

def check_normalize(y_values):
    if max(y_values) > 1:
        y_values = y_values / max(y_values)
    return y_values

# def save_processed_ir_csv(x_values, spectrum, csv_filename):
#     df = pd.DataFrame({"Wavenumber (cm⁻¹)": x_values, "Intensity": spectrum})
#     df.to_csv(csv_filename, index=False)
#     print(f"Done. The is spectrum is saved as {csv_filename}.")
    
def make_conv_matrix(frequencies=list(range(500,4001,1)),std_dev=10):
    length=len(frequencies)
    gaussian=[(1/(2*math.pi*std_dev**2)**0.5)*math.exp(-1*((frequencies[i])-frequencies[0])**2/(2*std_dev**2)) for i in range(length)]
    conv_matrix=np.empty([length,length])
    for i in range(length):
        for j in range(length):
            conv_matrix[i,j]=gaussian[abs(i-j)]
    return conv_matrix

def spectral_information_similarity(spectrum1, spectrum2,
                                    conv_matrix,
                                    frequencies=list(range(400,4002,2)),
                                    threshold=1e-10):
    """
    spectrum1: IR spectrum from Gaussian calculation
    spectrum2: IR spectrum from QM9S/QMe14S
    """
    length = len(spectrum1)
    assert length == len(spectrum2), f"compared spectra are of different lengths (spectrum1: {len(spectrum1)}, spectrum2: {len(spectrum2)})"
    assert length == len(frequencies), "compared spectra are a different length than the frequencies list, which can be specified"

    spectrum1 = check_normalize(spectrum1)
    spectrum2 = check_normalize(spectrum2)


    nan_mask=np.isnan(spectrum1)+np.isnan(spectrum2)
    spectrum1[spectrum1<threshold]=threshold
    spectrum2[spectrum2<threshold]=threshold
    spectrum1[nan_mask]=0
    spectrum2[nan_mask]=0

    spectrum1=np.expand_dims(spectrum1,axis=0)
    spectrum2=np.expand_dims(spectrum2,axis=0)

    conv1=np.matmul(spectrum1,conv_matrix)
    conv2=np.matmul(spectrum2,conv_matrix)
    conv1[0,nan_mask]=np.nan
    conv2[0,nan_mask]=np.nan

    sum1=np.nansum(conv1)
    sum2=np.nansum(conv2)
    norm1=conv1/sum1
    norm2=conv2/sum2
    distance=norm1*np.log(norm1/norm2)+norm2*np.log(norm2/norm1)
    sim=1/(1+np.nansum(distance))

    return sim

# def plot_spec(y_origin,
#               y_gaussian, 
#               x_origin=np.linspace(500,4000,3501), 
#               x_gaussian=np.linspace(500,4000,3501), 
#               title=None,
#               SIS=None,
#               save_name=None):
    
#     plt.figure(figsize=(8, 4))
#     plt.plot(x_gaussian, y_gaussian, label="Broadened IR Spectrum")
#     plt.plot(x_origin, y_origin, label="IR Spectrum from QM9S", alpha=0.7)
#     plt.xlabel("Wavenumber (cm⁻¹)")
#     plt.ylabel("Intensity")
#     if title is None:
#         title = "Infrared Spectrum Comparison"
#     if SIS is not None:
#         title += f" | SIS: {SIS:.2f}"
    
#     plt.title(title)
#     plt.gca().invert_xaxis()  # IR 频谱通常从高频到低频显示
#     plt.legend()
#     if save_name is None:
#         save_name = "spec_comparison.png"
#     plt.savefig(save_name)
#     plt.close()

# def find_ir_origin(smi, ir_df, x_origin, save_name='ir_origin.npz'):
#     if smi in ir_df['smiles']:
#         ir_origin = ir_df[ir_df['smiles']==smi]['spectra'].values[0]
#     else:
#         for smi_origin in ir_df['smiles']:
#             smi_canon = Chem.CanonSmiles(smi_origin)
#             if smi_canon == smi:
#                 ir_origin = ir_df[ir_df['smiles']==smi_origin]['spectra'].values[0]
#                 break
        
    
#     if save_name is not None:
#         np.savez(save_name, ir_origin=ir_origin, smi=smi, x_origin=x_origin)
#     return ir_origin

def find_log_file(dir_name):
    for file in os.listdir(dir_name):
        if file.endswith(".log"):
            return os.path.join(dir_name, file)
    return None

def log_2_ir_sis(log_file, ir_ref, x_values, conv_matrix):
    """
    args:
        log_file: gaussian log file including IR intensity
    """
    freq, inten = parse_gaussian_log(log_file)
    if type(freq) == list and freq == []:
        return None, None
    ir_computed = broaden_spectrum(frequencies=freq, intensities=inten, x_values=x_values)
    ir_computed = check_normalize(ir_computed)
    
    sis = spectral_information_similarity(ir_ref, ir_computed, conv_matrix, x_values)

    return ir_computed, sis

def cal_func_region_sis(ir_ref, ir_computed, origin_x=np.linspace(500, 4000, 3501)):
    x_func_region = np.linspace(1350, 4000, 2651)
    
    func_ref = interpolate.interp1d(origin_x, ir_ref, kind='slinear')
    ir_ref_func_region = func_ref(x_func_region)

    func_computed = interpolate.interp1d(origin_x, ir_computed, kind='slinear')
    ir_computed_func_region = func_computed(x_func_region)

    conv_matrix = make_conv_matrix(frequencies=x_func_region, std_dev=10)
    sis_func_region = spectral_information_similarity(ir_ref_func_region, ir_computed_func_region, 
                                                      conv_matrix, x_func_region)
    return sis_func_region

def check_struc_match(sample_smi, gaussianfile, bond_threshold):
    charges, xyz = log2mol(gaussianfile)
    if charges is not None:
        mol = XYZ2MOL(atom_charge=charges, pos=xyz, num_atoms=len(charges), output_log=False, bond_threshold=bond_threshold)
        mol, smi_not_canonical = mol.xyz2mol(process_fragments=False)
        try:
            smi_canon = get_clean_smi(mol)
        except Exception as e:
            print(f"Error: {e}")
            return False
        
        return check_quiv_by_inchikey(sample_smi, smi_canon)
    else:
        return False


def log2mol(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # check if terminate normally
    if not any("Normal termination of Gaussian" in line for line in lines[-20:]):
        print("Not normal termination of Gaussian!")
        return None, None

    start_indices = [i for i, line in enumerate(lines) if "Standard orientation:" in line]
    if not start_indices:
        print("There is no 'Standard orientation'.")
        return None, None
    # last Standard orientation
    start = start_indices[-1] + 5  

    charges = []
    xyz = []
    for line in lines[start:]:
        if '-----' in line or not line.strip():
            break  
        parts = line.split()
        charges.append(int(parts[1]))
        xyz.append([float(parts[3]), float(parts[4]), float(parts[5])])
    return np.array(charges), np.array(xyz)

def all_stable_cls_file(results_dir):
    print(f"results_dir: {results_dir}")
    files = os.listdir(results_dir)
    mol_files = {}
    for file in files:
        assert 'mol' in file and file.split('_')[1] == 'mol', f"file: {file}"
        mol_idx = int(file.split('_')[2])
        if mol_idx not in mol_files:
            mol_files[mol_idx] = []
        mol_files[mol_idx].append(file)
    return mol_files

def main_ir_sis(args):
    
    sample_result = pd.read_csv(args.structral_analysis_result)
    ir_compute_dir = args.ir_compute_dir
    if args.start_mol != -1 and args.end_mol != -1:
        assert args.start_mol <= args.end_mol, f"start_mol {args.start_mol} must be less than or equal to end_mol {args.end_mol}."
        num_mol = args.end_mol - args.start_mol + 1
    elif args.num_mol is None:
        num_mol = sample_result['mol_idx'].iloc[-1] + 1
    else:
        num_mol = args.num_mol
    num_sample = sample_result['sample_idx'].iloc[-1] + 1

    qm9s_ir_df = pd.read_pickle(args.test_data)
    

    if args.all_stable:
        mol_files = all_stable_cls_file(args.ir_compute_dir)
    
    ir_compute_dic = {'mol_idx':[],'smiles':[], 'ir_idx':[],
                      'similarity':[], 'stable_similarity':[], 'is_smi_matched':[], 
                      'sis':[], 'sis_func':[], 'ir':[] }
    if args.dataset == "qm9s": 
        bond_threshold = 0.4
        x_values = np.linspace(500, 4000, 3501)
    elif args.dataset == "qme14s": 
        bond_threshold = 0.3
        x_values = np.linspace(500, 4000, 3500)
  
    
    for i in tqdm(range(num_mol), desc="check tgt"):
        mol_idx = [i] * (num_sample + 2)
        smi_tgt = sample_result['smi_tgt'][i*num_sample]

        smi_origin = qm9s_ir_df.iloc[i]['smiles']
        if 'Al' in smi_origin:
            smi_origin = get_clean_smi(Chem.MolFromSmiles(smi_origin), ignore_metal_explicit_H=True)
            smi_tgt = get_clean_smi(Chem.MolFromSmiles(smi_tgt), ignore_metal_explicit_H=True)
        else:
            smi_origin = get_clean_smi(Chem.MolFromSmiles(smi_origin))
        if not check_quiv_by_inchikey(smi_origin, smi_tgt):
            raise ValueError(f"{i} | Converted smiles {smi_tgt} and smiles from {args.dataset} {smi_origin} do not match.")

    for i in tqdm(range(num_mol)):
        if args.start_mol != -1 and args.end_mol != -1:
            i = args.start_mol + i

        mol_idx = [i] * (num_sample + 2)
        ir_idx = ['qm9s', 'tgt'] + list(range(num_sample))
        ir = [None] * (num_sample + 2)
        sis = [None] * (num_sample + 2)
        sis_func = [None] * (num_sample + 2)

        smi_tgt = sample_result['smi_tgt'][i*num_sample]
        smi_sample = sample_result[sample_result["smi_tgt"]==smi_tgt]["smi_sampled"].to_list()
        smiles = [smi_tgt]*2 + smi_sample
        
        similarity = [None]*2 + sample_result[sample_result["smi_tgt"]==smi_tgt]["similarity"].to_list()
        stable_similarity = [None]*2 + sample_result[sample_result["smi_tgt"]==smi_tgt]["stable_similarity"].to_list()
        
        is_smi_matched = [None]*2 + sample_result[sample_result["smi_tgt"]==smi_tgt]["is_smi_matched"].to_list()

        
        if 'Al' in smi_origin:
            smi_tgt = get_clean_smi(Chem.MolFromSmiles(smi_tgt), ignore_metal_explicit_H=True)


        ir_qm9s = qm9s_ir_df.iloc[i]['spectra']
        ir_qm9s = check_normalize(ir_qm9s)
        ir[0] = ir_qm9s
        
        conv_matrix = make_conv_matrix(frequencies=x_values, std_dev=10)
        
        if args.only_stable_logs:
            dir_i = os.path.join(ir_compute_dir, f"{i}_all")
        else:
            dir_i = os.path.join(ir_compute_dir, f"mol_{i}")

        if (not (args.only_stable_logs or args.all_stable)):
            dir_i_tgt = os.path.join(dir_i, "tgt")
            log_file_tgt = find_log_file(dir_i_tgt)
            if log_file_tgt is not None:
                ir_tgt, sis_tgt = log_2_ir_sis(log_file_tgt, ir_ref=ir_qm9s, x_values=x_values, conv_matrix=conv_matrix)
                sis_tgt_func = cal_func_region_sis(ir_ref=ir_qm9s, ir_computed=ir_tgt, origin_x=x_values)
                ir[1] = ir_tgt
                sis[1] = sis_tgt
                sis_func[1] = sis_tgt_func
               
                  
        for j in tqdm(range(num_sample)):
            if args.only_stable_logs:
                log_file_j = os.path.join(dir_i, f"mol_{i}_sample_{j}.log")
                if not os.path.exists(log_file_j): log_file_j = None
            if args.all_stable:
                log_file_j = None
                if i not in mol_files: continue
                for dir_ij in mol_files[i]:
                    if dir_ij.endswith(f"sample_{j}"):
                        log_file_j = os.path.join(ir_compute_dir, f"{dir_ij}/mol_{i}_sample_{j}.log")
                        break
            else:
                dir_i_j = os.path.join(dir_i, f"sample_{j}")
                log_file_j = find_log_file(dir_i_j)

            if log_file_j is not None:
                
                if type(smi_sample[j]) != str: 
                    print(f"mol {i}, sample {j} is invalid")
                    continue # invalid molecule
                if type(stable_similarity[j+2]) is None: 
                    print(f"mol {i}, sample {j} is unstable")
                    continue # unstable molecule
                if not check_struc_match(smi_sample[j], log_file_j, bond_threshold):  
                    print(f"mol {i}, sample {j} not match")
                    continue

                ir_j, sis_j = log_2_ir_sis(log_file_j, ir_ref=ir_qm9s, x_values=x_values, conv_matrix=conv_matrix)
                sis_j_func = cal_func_region_sis(ir_ref=ir_qm9s, ir_computed=ir_j, origin_x=x_values)
                ir[j+2] = ir_j
                sis[j+2] = sis_j
                sis_func[j+2] = sis_j_func

        
        ir_compute_dic['mol_idx'] += mol_idx
        ir_compute_dic["smiles"] += smiles
        ir_compute_dic['ir_idx'] += ir_idx
        ir_compute_dic["similarity"] += similarity
        ir_compute_dic["stable_similarity"] += stable_similarity
        ir_compute_dic['is_smi_matched'] += is_smi_matched
        
        ir_compute_dic["sis"] += sis
        ir_compute_dic['sis_func'] += sis_func
        ir_compute_dic['ir'] += ir

        # break
    ir_compute_df = pd.DataFrame(ir_compute_dic)
    file_name = os.path.join(ir_compute_dir, f"ir_computed_sis_{num_mol}.pkl")
    ir_compute_df.to_pickle(file_name)
    print(ir_compute_df)
    print(f'Done. ir_compute_df is saved to {file_name}')
    compute_df = ir_compute_df.drop(columns=['ir'])
    compute_df.to_csv(os.path.join(ir_compute_dir, f"computed_sis_{num_mol}.csv"))


# def plot_sis_tsim(tsim, sis):
#     sns.scatterplot(x=tsim, y=sis)
#     plt.xlabel('Tanimoto Similarity')
#     plt.ylabel('Spectral Information Similarity')
#     plt.savefig('./tsim_sis.png', dpi=800)
#     plt.close()

#     sns.histplot(sis, bins=50)
#     plt.xlabel('Spectral Information Similarity')
#     plt.ylabel('Frequency')
#     plt.savefig('./sis.png', dpi=800)
#     plt.close()

#     sns.histplot(tsim, bins=50)
#     plt.xlabel('Tanimoto Similarity')
#     plt.ylabel('Frequency')
#     plt.savefig('./tsim.png', dpi=800)
#     plt.close()

# def retrieve_comparison_plots(spec_sis_df, src_dir):
#     save_dir = "./comparison_plots"
#     sis_min = 0.2
#     step = 0.1
#     while sis_min < 1:
#         sis_max = sis_min + 0.1
#         save_dir_i = os.path.join(save_dir, f"{sis_min}_{sis_max}")
#         if os.path.exists(save_dir_i):
#             os.mkdir(save_dir_i)
#             spec_sis_df_i = spec_sis_df[spec_sis_df["sis"]>sis_min]
#             spec_sis_df_i = spec_sis_df_i[spec_sis_df_i["sis"]<sis_max]
#             for j, row in spec_sis_df_i.iterrows():
#                 mol_idx = row["mol_idx"]
#                 ir_idx = row['ir_idx']
#                 t_sim = row['tanimoto_similarity']
#                 is_smi_matched = row['is_smi_matched']
#                 if ir_idx == 'tgt':
#                     new_fig_name = f"mol{mol_idx}_{ir_idx}.png"
#                     shutil.copy(os.path.join(src_dir, f'mol_{mol_idx}/tgt/ir_comparison.png'),
#                                 os.path.join(save_dir_i, new_fig_name))
#                 else:
#                     new_fig_name = f"mol{mol_idx}_sample{ir_idx}_{t_sim:2f}_{is_smi_matched}.png"
#                     shutil.copy(os.path.join(src_dir, f'mol_{mol_idx}/sample_{ir_idx}/ir_comparison.png'),
#                                 os.path.join(save_dir_i, new_fig_name))



def compute_mean_max_sis(spec_sis_df):

    sis_sample_mean = spec_sis_df['sis'].mean()
    sis_func_sample_mean = spec_sis_df['sis_func'].mean()
    print(f"Average SIS of samples | all: {sis_sample_mean:.3f} | functional group region: {sis_func_sample_mean:.3f}")

    sis_sample_max = []
    sis_func_sample_max = []
    is_match = []
    num_mol = spec_sis_df['mol_idx'].iloc[-1] + 1
    for i in tqdm(range(num_mol)):
        df_i = spec_sis_df[spec_sis_df['mol_idx']==i]
        sis_i = df_i['sis']
        
        sis_sample_max.append(sis_i.max())
        sis_func_sample_max.append(df_i['sis_func'].max())
      
        is_match_i = int(df_i['is_smi_matched'].sum() >= 1)
        is_match.append(is_match_i)
    sis_sample_max_mean = np.nanmean(sis_sample_max)
    sis_func_sample_max_mean = np.nanmean(sis_func_sample_max)
    mol_acc = np.mean(is_match)

    return sis_sample_mean, sis_func_sample_mean, sis_sample_max_mean, sis_func_sample_max_mean, mol_acc


    
def main_analyze_sis(args):
    spec_sis_df = pd.read_pickle(args.cal_sis_result)
    save_path = os.path.dirname(args.cal_sis_result)
    
    if args.log_file_name == '': log_file_name='spectral_analysis.log'
    elif '.log' not in args.log_file_name: log_file_name = args.log_file_name + '.log'

    log_file_name = os.path.join(save_path, log_file_name)
    print(f"Log file: {log_file_name}")
    logging.basicConfig(filename=log_file_name,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing sample results: {args.cal_sis_result}")

    
    if args.random and type(args.num_mol) == int:
        num_mol_all = spec_sis_df.iloc[-1]["mol_idx"] + 1
        mol_idx_list = np.random.randint(0, num_mol_all, size=args.num_mol).tolist()
        logger.info(f"Number of test data: {args.num_mol} | Random mol idx: {mol_idx_list}")
        spec_sis_df = spec_sis_df[spec_sis_df["mol_idx"].isin([i for i in range(args.num_mol)])]

        sis_result = analyze_sis(spec_sis_df, logger)

    elif type(args.num_mol) == int:
        logger.info(f"Number of test data: {args.num_mol}")
        spec_sis_df = spec_sis_df[spec_sis_df["mol_idx"].isin([i for i in range(args.num_mol)])]

        sis_result = analyze_sis(spec_sis_df, logger)

    elif type(args.num_mol) == list:
        sis_result = {
            'num_mol': args.num_mol,
            "sis_sample_mean_noNan": [], "sis_func_sample_mean_noNan": [],
            "sis_sample_mean_stable": [], "sis_func_sample_mean_stable": []
        }
        for n in args.num_mol:
            logger.info(f"Number of test data: {n}")
            mol_idx_list = [i for i in range(n)]
            spec_sis_df_n = spec_sis_df[spec_sis_df["mol_idx"].isin(mol_idx_list)]
            sis_result_n = analyze_sis(spec_sis_df_n, logger)
            sis_result['sis_sample_mean_noNan'].append(sis_result_n['sis_sample_mean_noNan'])
            sis_result['sis_func_sample_mean_noNan'].append(sis_result_n['sis_func_sample_mean_noNan'])
            sis_result['sis_sample_mean_stable'].append(sis_result_n['sis_sample_mean_stable'])
            sis_result['sis_func_sample_mean_stable'].append(sis_result_n['sis_func_sample_mean_stable'])
        df = pd.DataFrame(sis_result)
        df.to_csv(os.path.join(save_path, 'sis_result.csv'))

def analyze_sis(spec_sis_df, logger):

    sis_repro = spec_sis_df[spec_sis_df['ir_idx']=='tgt']['sis'].mean()
    sis_func_repro = spec_sis_df[spec_sis_df['ir_idx']=='tgt']['sis_func'].mean()
    logger.info(f"Average SIS of reproduction | all: {sis_repro:.3f} | functional group region: {sis_func_repro:.3f}")

    # remove none ir which has been removed when computing sis
    sis_na_df = spec_sis_df['sis'].isna()
    spec_sis_df = spec_sis_df[~sis_na_df]
    
    # remove unfinished ir 
    logger.info("Removing unfinished ir spectra...")
    target_value = 1e-10
    unfinished_ir_df = spec_sis_df[spec_sis_df['ir'].apply(lambda x: np.allclose(x, target_value))]
    spec_sis_df_noNan = spec_sis_df.drop(unfinished_ir_df.index)
    
    # remove qm9s, tgt  
    spec_sis_df_noNan = spec_sis_df_noNan.dropna(subset=['sis', "similarity"])

    sis_sample_mean_noNan, sis_func_sample_mean_noNan, sis_sample_max_mean_noNan, sis_func_sample_max_mean_noNan, mol_acc_noNan = compute_mean_max_sis(spec_sis_df_noNan)
    logger.info(f"Mol acc: {mol_acc_noNan:.3f}")
    logger.info(f"Average SIS of samples | all: {sis_sample_mean_noNan:.3f} | functional group region: {sis_func_sample_mean_noNan:.3f}")
    logger.info(f"Max SIS of samples | all: {sis_sample_max_mean_noNan:.3f} | functional group region: {sis_func_sample_max_mean_noNan:.3f}")

    unstable_smi = []
    for i, row in spec_sis_df_noNan.iterrows():
        ss = row["stable_similarity"]
        if ss is None: unstable_smi.append(row['smiles'])

    # remove unstable mol
    stable_spec_df = spec_sis_df_noNan[~spec_sis_df_noNan['smiles'].isin(unstable_smi)]
    sis_sample_mean_stable, sis_func_sample_mean_stable, sis_sample_max_mean_stable, sis_func_sample_max_mean_stable, mol_acc_stable = compute_mean_max_sis(stable_spec_df)
    logger.info(f"Stable Mol: Mol acc: {mol_acc_stable:.3f}")
    logger.info(f"Stable Mol: Average SIS of samples | all: {sis_sample_mean_stable:.3f} | functional group region: {sis_func_sample_mean_stable:.3f}")
    logger.info(f"Stable Mol: Max SIS of samples | all: {sis_sample_max_mean_stable:.3f} | functional group region: {sis_func_sample_max_mean_stable:.3f}")
    return {
        "sis_sample_mean_noNan": sis_sample_mean_noNan, "sis_func_sample_mean_noNan": sis_func_sample_mean_noNan,
        "sis_sample_mean_stable": sis_sample_mean_stable, "sis_func_sample_mean_stable": sis_func_sample_mean_stable
    }

def main_analyze_sis_report(args):
    spec_sis_df = pd.read_pickle(args.cal_sis_result)
    save_path = os.path.dirname(args.cal_sis_result)
    
    if args.log_file_name == '': log_file_name='spectral_analysis.log'
    elif '.log' not in args.log_file_name: log_file_name = args.log_file_name + '.log'

    log_file_name = os.path.join(save_path, log_file_name)
    print(f"Log file: {log_file_name}")
    logging.basicConfig(filename=log_file_name,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing sample results: {args.cal_sis_result}")

    if type(args.num_mol) == int:
        logger.info(f"Number of test data: {args.num_mol}")
        spec_sis_df = spec_sis_df[spec_sis_df["mol_idx"].isin([i for i in range(args.num_mol)])]


    



if __name__ == "__main__":
    parser = ArgumentParser()
    

    parser.add_argument('--cal_sis', action="store_true")
    parser.add_argument('--only_stable_logs', action="store_true")
    parser.add_argument('--all_stable', action="store_true")
    parser.add_argument('--structral_analysis_result', type=str, help=".csv file")
    parser.add_argument('--ir_compute_dir', type=str, help="directory")
    parser.add_argument('--test_data', type=str, help=".pkl file")

    parser.add_argument('--num_mol', type=eval, default=None)
    parser.add_argument('--start_mol', type=int, default=-1)
    parser.add_argument('--end_mol', type=int, default=-1)

    parser.add_argument('--cal_sis_result', type=str, default='')
    parser.add_argument('--log_file_name',type=str, default='')
    parser.add_argument('--random', action="store_true")

    parser.add_argument('--dataset', type=str, help="qme14s or qm9s")



    args = parser.parse_args()

    if args.cal_sis:
        main_ir_sis(args)
    elif args.cal_sis_result != '':
        main_analyze_sis_report(args)

