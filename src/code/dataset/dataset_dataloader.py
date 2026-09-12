import torch
from torch.utils.data import Dataset, DataLoader

import os
import os.path as osp
import pandas as pd
import numpy as np
from scipy import interpolate

import warnings
warnings.filterwarnings("error", category=RuntimeWarning)



qm9s_dataset_info = {
    'name': 'qm9s',
    'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
    'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
    'max_n_nodes': 29
    
}


qme14s_dataset_info = {
    'name': 'qme14s',
    'atom_encoder': {'H': 0, 'B':1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'Al': 6, 'Si': 7, 'P': 8, 'S': 9, 'Cl': 10, 'As': 11, 'Se': 12, 'Br': 13},
    'atom_decoder': ['H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Se','Br'],
}



def prep_ir_geo_dataset(data_dir, dataset, split_file='', test_run=False, train_include_qm9s=False):
    
   
    dataset_dic = {"train":None, "val":None, "test":None}

    if split_file != '':
        df_all = pd.read_pickle(osp.join(data_dir, split_file))
        df_all = df_all.sample(frac=1)
        print(df_all)
        
        test_df = df_all.iloc[:1000]
        dataset_dic["test"] = IRGeoDataset(test_df, dataset=dataset)
        test_df.to_pickle(osp.join(data_dir, "test_1000_qm9s.pkl"))
        
        train_val_df = df_all.iloc[1000:]
        num_train = int(0.95*len(train_val_df))
        train_df = train_val_df.iloc[:num_train]
        dataset_dic["train"] = IRGeoDataset(train_df, dataset=dataset)
        train_df.to_pickle(osp.join(data_dir, f"train_{num_train}_qm9s.pkl"))
        
        val_df = train_val_df.iloc[num_train:]
        dataset_dic["val"] = IRGeoDataset(val_df, dataset=dataset)
        val_df.to_pickle(osp.join(data_dir, f"val_{len(val_df)}_qm9s.pkl"))


    else:
        print("No new datasets are splitted.")
        files = os.listdir(data_dir)
        if test_run: 
            for file_name in files:
                if "val_" in file_name: break
            return IRGeoDataset(pd.read_pickle(osp.join(data_dir,file_name)))

        for file_name in files:
            if "train_" in file_name and '.pkl' in file_name:
                if train_include_qm9s and ("include_qm9s" not in file_name): continue
                if not train_include_qm9s and ("include_qm9s" in file_name): continue
                train_df = pd.read_pickle(osp.join(data_dir,file_name))
                print("train data num:", len(train_df))
                dataset_dic["train"] = IRGeoDataset(train_df, dataset=dataset)
            elif "val_" in file_name and '.pkl' in file_name:
                val_df = pd.read_pickle(osp.join(data_dir,file_name))
                dataset_dic["val"] = IRGeoDataset(val_df, dataset=dataset)
            elif "test_" in file_name and '.pkl' in file_name:
                test_df = pd.read_pickle(osp.join(data_dir,file_name))
                dataset_dic["test"] = IRGeoDataset(test_df, dataset=dataset)
    
    return dataset_dic



def process_spec(origin_y, origin_x):
    new_x = np.linspace(500, 4000, 3200)
    intp_func = interpolate.interp1d(origin_x, origin_y)
    new_y = intp_func(new_x)

    # normalization of ir spectra
    new_y = new_y / np.max(new_y) * 99
        
    new_y_int = new_y.astype(int)
    new_y_int = np.clip(new_y_int, 0, 99)

    return new_y_int


class IRGeoDataset(Dataset):
    def __init__(self, data_df, dataset='qm9s'):
        """
            data_df: ['positions', 'charges', 'num_atoms', 'spectra', 'smiles']
            
        """
        self.data_df = data_df
        # print(data_df.columns)
        self.dataset = dataset

        if dataset == 'qm9s': 
            self.included_species = torch.tensor([1, 6, 7, 8, 9]) # H, C, N, O, F
            self.origin_x = np.linspace(500, 4000, 3501)

        elif dataset == 'qme14s': 
            # 'H', 'B', 'C', 'N', 'O', 'F', 'Al', 'Si', 'P', 'S', 'Cl', 'As', 'Se','Br'
            self.included_species = torch.tensor([1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 33, 34, 35]) 
            self.origin_x = np.linspace(500, 4000, 3500)


    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        data_i = self.data_df.iloc[idx]

        processed_spec = process_spec(data_i["spectra"], origin_x=self.origin_x)
   
        data_i_ = {key: torch.tensor(value) if key not in ["smiles", "spectra", "clean_smiles"] else value for key, value in data_i.to_dict().items() }
        
        data_i_["spectra"] = torch.from_numpy(processed_spec)
        
        data_i_["one_hot"] = (data_i_["charges"].unsqueeze(-1) == self.included_species.unsqueeze(0).unsqueeze(0)).int()
        return data_i_

def get_edge_mask(atom_mask, batch_size, max_num_atoms):
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_num_atoms * max_num_atoms, 1)
    return edge_mask

def dataset_collate_fn(data, dataset):
    """
    collate_fn receives a list of tuples if your __getitem__ function from a Dataset.
    data:
        a list of dictionary.
    """
    # print(data)
    num_atoms = torch.stack([data_i["num_atoms"] for data_i in data])
    max_num_atoms = max(num_atoms)
    batch_size = num_atoms.size(0)
    
    include_formula = False
    if 'formula' in data[0]:
        include_formula = True
        max_len_formula = max(data_i['formula'].shape[0] for data_i in data )
        formula = torch.zeros(batch_size, max_len_formula)
    else:
        formula = torch.zeros(batch_size, 1)

    positions = torch.zeros(batch_size, max_num_atoms, 3)
    charges = torch.zeros(batch_size, max_num_atoms)
    if dataset == 'qm9s':
        one_hot = torch.zeros(batch_size, max_num_atoms, 5)

    elif dataset == 'qme14s':
        one_hot = torch.zeros(batch_size, max_num_atoms, 14)

    for i in range(batch_size):
        if include_formula:
            formula[i, :data[i]['formula'].shape[0]] =  data[i]['formula']

        positions[i, :num_atoms[i], :] = data[i]["positions"]
        charges[i, :num_atoms[i]] = data[i]["charges"]
        one_hot[i, :num_atoms[i], :] = data[i]["one_hot"]
        
    
    atom_mask = (charges > 0)
    edge_mask = get_edge_mask(atom_mask, batch_size, max_num_atoms)
    spectra = torch.stack([data_i["spectra"] for data_i in data])
    

    if 'func_groups' in data[0]:
        fg = torch.stack([data_i["func_groups"] for data_i in data])
    else:
        fg = torch.zeros(batch_size, 1)
    

   

    return {
        "positions": positions,
        "charges": charges.unsqueeze(-1),
        "num_atoms": num_atoms,
        "spectra": spectra,
        "smiles": [data_i["smiles"] for data_i in data],
        "atom_mask": atom_mask,
        "edge_mask": edge_mask,
        "one_hot": one_hot,
        "func_groups": fg,
        "formula": formula
        }
    

def qm9s_dataset_collate_fn(data):
    return dataset_collate_fn(data, dataset='qm9s')

def qme14s_dataset_collate_fn(data):
    return dataset_collate_fn(data, dataset='qme14s')


def get_dataloaders(data_mode, batch_size, shuffle=False, num_workers=0, data=None, test_run=False):
    if data_mode == "raw":
        if test_run: return prep_ir_geo_dataset(test_run=test_run)
        dataset_dic = prep_ir_geo_dataset()
    elif data_mode == "dataset":
        assert data is not None
        dataset_dic = data

    dataloaders = {}
    for key in dataset_dic.keys():
        dataloaders[key] = DataLoader(dataset_dic[key],
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      collate_fn=dataset_collate_fn
                                     )
    return dataloaders
    




if __name__ == "__main__":
    dataset = prep_ir_geo_dataset()
    print(dataset[0])
    loader = DataLoader(dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=dataset_collate_fn
            )
    for batch in loader:
        print(batch)
        break
    