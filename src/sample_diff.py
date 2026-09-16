import torch
from torch.utils.data import DataLoader, Subset


from .code.dataset.dataset_dataloader import (prep_ir_geo_dataset, 
                                         qm9s_dataset_collate_fn,
                                         qme14s_dataset_collate_fn,
                                         )
from .code.get_model import get_checkpoint, load_diffusion
from .code.model.model_utils import (assert_correctly_masked,
                                assert_mean_zero_with_mask)
from .code.model.spec_fg_cls import SpecFuncGroupsClsModel

from argparse import ArgumentParser
from pathlib import Path
import os.path as osp
import os
from tqdm import tqdm
import numpy as np



def sample_chain(spec, one_hot, charges, node_mask, edge_mask, 
                 generative_model,
                 formula_ids=None
                 ):
  

    batch_size, max_n_nodes, _ = node_mask.shape
    h = {'categorical': one_hot, 'integer': charges}
    
    chain = generative_model.sample_chain(spec=spec,
                                            h = h,
                                            n_samples=batch_size, 
                                            n_nodes=max_n_nodes,
                                            node_mask=node_mask,
                                            edge_mask=edge_mask,
                                            context=None,
                                            formula_ids=formula_ids
                                            )
    
    chain_bacth_first = chain.permute(1, 0, 2, 3) # (batch_size, frames, max_n_nodes, n_dims)

    return chain_bacth_first 
 

def sample_ff(spec, one_hot, charges, node_mask, edge_mask, 
           generative_model,
           fix_noise: bool=False,
           sample_mode="sample", sample_times=1,
           formula_ids=None):
 
    context = None

    batch_size, max_n_nodes, _ = node_mask.shape
    h = {'categorical': one_hot, 'integer': charges}
    
    x_all_sampled = torch.zeros(
        batch_size, sample_times, max_n_nodes, 3, device=node_mask.device)
    for i in tqdm(range(sample_times), desc="sample_times"):
        x = generative_model.sample(spec=spec,
                                    h = h,
                                    n_samples=batch_size, 
                                    n_nodes=max_n_nodes,
                                    node_mask=node_mask,
                                    edge_mask=edge_mask,
                                    context=context,
                                    fix_noise=fix_noise,
                                    sample_mode=sample_mode,
                                    formula_ids=formula_ids
                                    )
        assert_correctly_masked(x, node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        x_all_sampled[:, i, :, :] = x


    return x_all_sampled



def get_checkpoint_diff(exp_dir_path, last_checkpoint=False, checkpoint=None):
    files = os.listdir(exp_dir_path)
    for f in files:
        if '.ckpt' in f: 
            if checkpoint is not None:
                if (checkpoint in f): break
            elif last_checkpoint and ('last' in f): break
            elif not last_checkpoint and ('last' not in f): break
            # elif (not last_checkpoint) 
                # break
    return osp.join(exp_dir_path, f), f



def stack_array(array_list, sample_times=1):
    shape_len = len(array_list[0].shape)
    if shape_len == 4:
        max_num_nodes = max(array_i.shape[2] for array_i in array_list)
    else:
        max_num_nodes = max(array_i.shape[1] for array_i in array_list)
    
    padded_arr_list = []
    for array in array_list:
        if shape_len == 4:
            bs, sample_times, nodes, dims = array.shape
            padded_arr = np.zeros((bs, sample_times, max_num_nodes, dims))
            padded_arr[:, :, :nodes, :] = array
        elif shape_len == 3:
            bs, nodes, dims = array.shape
            padded_arr = np.zeros((bs, max_num_nodes, dims))
            padded_arr[:, :nodes, :] = array
        elif shape_len == 2:
            bs, nodes = array.shape
            padded_arr = np.zeros((bs, max_num_nodes))
            padded_arr[:, :nodes] = array
        padded_arr_list.append(padded_arr)

    
    stacked_array = np.vstack(padded_arr_list)
    # print("stacked_array", stacked_array.shape)
    return stacked_array

def main(args):
   
    dataset = prep_ir_geo_dataset(args.data_dir, args.dataset)

    test_dataset = dataset["test"]
   
    
    if args.data_num is not None:
     
        indices = [i for i in range(args.data_num)]
        test_dataset = Subset(test_dataset, indices)
    elif args.sample_mol_idx is not None:
        print("sample_mol_idx", args.sample_mol_idx)
        test_dataset = Subset(test_dataset, args.sample_mol_idx)
        
    if args.dataset == "qm9s":
        test_loader = DataLoader(test_dataset, 
                                collate_fn=qm9s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
    elif args.dataset == "qme14s":
        test_loader = DataLoader(dataset["test"], 
                                collate_fn=qme14s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    device = torch.device("cuda") if (args.cuda and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float32
    
    diff_model, diff_checkpoint_name, diff_args = load_diffusion(diff_dir_path=args.diff_dir_path, 
                                                                 device=device,
                                                                 last_checkpoint=args.last_checkpoint,
                                                                 checkpoint=args.checkpoint)
                                                                #  use_full_cls=args.use_full_cls)
    diff_model = diff_model.to(device)
    diff_model.eval()

    pos_pred_list = []
    pos_tgt_list = []
    onehot_list = []
    charges_list = []
    node_mask_list = []
    n_test_data = 0

    

    for batch in tqdm(test_loader):
        pos_tgt_list.append(batch['positions'].numpy())

        one_hot = batch['one_hot']
        charges = (batch['charges'] if diff_args.include_charges else torch.zeros(0))
        onehot_list.append(one_hot.cpu().numpy())
        charges_list.append(charges.cpu().numpy())

        node_mask = batch['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = batch['edge_mask'].to(device, dtype)
        spec = batch["spectra"].to(device, dtype)

        if diff_args.use_fg or diff_args.use_formula:
            formula = batch["formula"].int().to(device)
        else: formula = None
        
        if diff_args.use_fg:
            fg = batch["func_groups"].to(device)
        
        n_test_data += spec.shape[0]

       

        if args.sample_chain:
            x = sample_chain(
                spec, 
                one_hot.to(device, dtype), 
                charges.to(device, dtype),
                node_mask, edge_mask, 
                generative_model=diff_model, 
                formula_ids=formula)
                                 
        else:
            x = sample_ff(
                spec, one_hot.to(device, dtype), 
                charges.to(device, dtype),
                node_mask, edge_mask, 
                generative_model=diff_model, 
                fix_noise=False,
                sample_mode=args.sample_mode,
                sample_times=args.sample_times,
                formula_ids=formula)
        
                                        
        node_mask_list.append(node_mask.squeeze(2).cpu().numpy())
        pos_pred_list.append(x.cpu().numpy())
        
        if args.test_sample:
            break
    
    if args.save_name == "None":
        save_name = diff_checkpoint_name.split('.')[0]
    else:
        save_name = args.save_name

    save_name = f"{save_name}_N{n_test_data}_t{args.sample_times}"
    
    if args.sample_chain:
        save_name = f"sample_chain_{save_name}.npz"
    else:
        save_name = f"sample_final_{save_name}"
        if args.sample_mode == "sample":
            save_name += '.npz'
        elif args.sample_mode == "only_mean":
            save_name += '_onlymean.npz'

    print("test_dataset", len(test_dataset))
    if len(test_dataset) >= 1000:
        save_path = osp.join(args.diff_dir_path, "sample")
    else:
        save_path = osp.join(args.diff_dir_path, "sample_test")
    if not osp.exists(save_path): os.mkdir(save_path)
    save_path = osp.join(save_path, f"N{n_test_data}_t{args.sample_times}")
    if not osp.exists(save_path): os.mkdir(save_path)
    save_file = osp.join(save_path, save_name)
    print(save_file)
    
    
    np.savez(save_file,
            pos_pred=stack_array(array_list=pos_pred_list, sample_times=args.sample_times),
            pos_tgt=stack_array(array_list=pos_tgt_list),
            onehot=stack_array(onehot_list),
            charges=stack_array(charges_list),
            node_mask=stack_array(node_mask_list)
            )



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--save_name", type=str, default=None)
    # parser.add_argument("--use_full_cls", action="store_true")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], required=True)
    parser.add_argument("--diff_dir_path", type=Path, default=True)
    parser.add_argument("--last_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument('--test_sample', action="store_true",
                        help="If True, only sample one batch.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=17)
    parser.add_argument("--cuda", type=eval, default=True)
    parser.add_argument("--sample_mode", type=str, default="sample", choices=["sample", "only_mean"])
    parser.add_argument("--sample_chain", action="store_true")
    parser.add_argument("--sample_times", type=int, default=1, 
                        help="the sampling number for each test data.")
    parser.add_argument("--data_num", type=eval, default=None, 
                        help="the number of test data")
    parser.add_argument("--sample_mol_idx", type=int, nargs='+', default=None)

    
    args = parser.parse_args()
    main(args)

    
 