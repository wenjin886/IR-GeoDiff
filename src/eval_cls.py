import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MultilabelAccuracy, MultilabelF1Score

import numpy as np
import random
from tqdm import tqdm
from argparse import ArgumentParser
import os
import os.path as osp
from pathlib import Path
import logging

from .code.model.spec_fg_cls import SpecFuncGroupsClsModel
from .code.dataset.dataset_dataloader import (
    prep_ir_geo_dataset, 
    qm9s_dataset_collate_fn, 
    qme14s_dataset_collate_fn)

from .code.get_model import get_checkpoint
from .code.dataset.qm9s_fg import fg_name_list as qm9s_fg_name_list
from .code.dataset.qme14s_fg import fg_name_list as qme14s_fg_name_list

from .code.model.model_utils import (
                           remove_mean_with_mask,
                           assert_mean_zero_with_mask,
                           )
from .code.model.train_utils import (check_mask_correct)

def set_seed(seed, device, num_gpu=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        if num_gpu > 1:
            torch.cuda.manual_seed_all(seed)  

def count_correct_mol(logits, fg_onehot):
    mol_acc = 0
    for pred, tgt in zip(logits, fg_onehot):
        if torch.all((pred > 0.5).int() == tgt): mol_acc += 1
    return mol_acc

def main(args):
    cls_checkpoint, checkpoint_name = get_checkpoint(args.cls_dir_path, last_checkpoint=args.last_checkpoint)
    
    if args.log_save_dir is not None:
        os.makedirs(args.log_save_dir, exist_ok=True)
        log_save_path = args.log_save_dir
    else:
        log_save_path = args.cls_dir_path
    print(f"log_save_path: {log_save_path}")
    raise NotImplementedError("Please specify log_save_dir to save the evaluation log.")
    log_save_path = osp.join(log_save_path, f"{checkpoint_name.split('.')[0]}_eval.log")

    logging.basicConfig(filename=log_save_path,  #
    # osp.join(args.cls_dir_path, f"{checkpoint_name.split('.')[0]}_eval.log"),
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Loading checkpoint from {cls_checkpoint}")

    
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    set_seed(seed=args.seed, device=device, num_gpu=args.num_gpu)
    if torch.cuda.is_available():
        logger.info(f"Current CUDA seed: {torch.cuda.initial_seed()}")
    torch.set_float32_matmul_precision(args.precision)


    # dataset = prep_qm9s_dataset(args.data_dir)
    dataset = prep_ir_geo_dataset(args.data_dir, args.dataset)

    test_dataset = dataset["test"]
    logger.info(f"test_dataset: {len(test_dataset)}")
   
    if args.dataset == "qm9s":
        
        test_loader = DataLoader(test_dataset, 
                                collate_fn=qm9s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
  
    elif args.dataset == "qme14s":
       
        test_loader = DataLoader(test_dataset, 
                                collate_fn=qme14s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # if args.eval_mode == "spec":
    classifier = SpecFuncGroupsClsModel.load_from_checkpoint(cls_checkpoint)
        
 
    
    classifier.to(device).eval() 
    dtype = torch.float32
    acc_per_fg = MultilabelAccuracy(num_labels=args.num_fg_cls, average="none").to(device)
    f1_per_fg = MultilabelF1Score(num_labels=args.num_fg_cls, average="none").to(device)

    n_test_data = 0
    cls_acc = torch.zeros(args.num_fg_cls, ).to(device)
    cls_f1 = torch.zeros(args.num_fg_cls, ).to(device)
    num_corr_mol = 0
    for batch in tqdm(test_loader, desc="batch"):
        
        fg_onehot = batch["func_groups"].to(device)
        bs = fg_onehot.shape[0]

        # if args.eval_mode == "spec":
        spectra = batch["spectra"].to(device)
        formula_ids = batch["formula"].to(device).int()
        with torch.no_grad():
            logits, _ = classifier(spectra, formula_ids)
            logits = logits.sigmoid()
            

        # elif args.eval_mode == "geo":
            
        #     spectra = batch["spectra"].to(device)
        #     x = batch['positions'].to(device, dtype)
        #     one_hot = batch['one_hot'].to(device, dtype)
        #     charges = batch['charges'].to(device, dtype)

        #     node_mask = batch['atom_mask'].to(device, dtype).unsqueeze(2)
        #     edge_mask = batch['edge_mask'].to(device, dtype)

        #     x = remove_mean_with_mask(x, node_mask)
        #     check_mask_correct([x, one_hot, charges], node_mask)
        #     assert_mean_zero_with_mask(x, node_mask)
        #     h = {'categorical': one_hot, 'integer': charges}
        #     z_x, z_h, _ = classifier.get_lantent_x_h(classifier.diff_model, spectra, x, h, node_mask, edge_mask)
        #     with torch.no_grad():
        #         logits = classifier(z_x, z_h, node_mask, edge_mask).sigmoid()
        
        n_test_data += bs
        
        cls_acc += acc_per_fg(logits, fg_onehot) * bs
        cls_f1 += f1_per_fg(logits, fg_onehot) * bs
        num_corr_mol += count_correct_mol(logits, fg_onehot)

    logger.info(f"test data: {n_test_data}")
    cls_acc = cls_acc/n_test_data
    cls_f1 = cls_f1/n_test_data

    if args.dataset == "qm9s": fg_name_list = qm9s_fg_name_list
    elif args.dataset == "qme14s": fg_name_list = qme14s_fg_name_list

    fg_name_max_len = max(len(fg_name) for fg_name in fg_name_list)
    for i in range(len(fg_name_list)):
        logger.info(f"{fg_name_list[i]:<{fg_name_max_len}}: {cls_acc[i]:.4f} | {cls_f1[i]:.4f}")
    logger.info(f"mol acc: {round(num_corr_mol/n_test_data, 3)}")
    

    



if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--eval_mode", choices=["spec", "geo"] )
    parser.add_argument("--cls_dir_path", type=str)
    parser.add_argument("--last_checkpoint", action="store_true", default=False)
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], required=True)
    parser.add_argument("--num_fg_cls", type=int)
    parser.add_argument("--data_dir", type=str)
    # parser.add_argument("--diff_dir_path", type=Path, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=17)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    parser.add_argument('--log_save_dir', type=str)
    
    args = parser.parse_args()
    print(args.log_save_dir)
    main(args)