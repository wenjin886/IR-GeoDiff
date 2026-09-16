import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pickle
import os

import numpy as np
import random


from .code.model.spec_fg_cls import SpecFuncGroupsClsModel
from .code.dataset.dataset_dataloader import (
    prep_ir_geo_dataset, 
    qm9s_dataset_collate_fn, 
    qme14s_dataset_collate_fn)

def set_seed(seed, device, num_gpu=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        if num_gpu > 1:
            torch.cuda.manual_seed_all(seed) 

def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M-%m-%d-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name

def main(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    set_seed(seed=args.seed, device=device, num_gpu=args.num_gpu)
    if torch.cuda.is_available():
        print(f"Current CUDA seed: {torch.cuda.initial_seed()}")

    if args.resume:
        assert args.checkpoint is not None and os.path.exists(args.checkpoint), f"Please check diff_checkoint: {args.checkpoint}."
    
    dataset = prep_ir_geo_dataset(args.data_dir, args.dataset, train_include_qm9s=args.train_include_qm9s)

    if args.dataset == "qm9s":
        train_loader = DataLoader(dataset["train"], 
                                  collate_fn=qm9s_dataset_collate_fn,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(dataset["val"], 
                                collate_fn=qm9s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        test_loader = DataLoader(dataset["test"], 
                                collate_fn=qm9s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    elif args.dataset == "qme14s":
        train_loader = DataLoader(dataset["train"], 
                                  collate_fn=qme14s_dataset_collate_fn,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        val_loader = DataLoader(dataset["val"], 
                                collate_fn=qme14s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)
        test_loader = DataLoader(dataset["test"], 
                                collate_fn=qme14s_dataset_collate_fn,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)


    
    
    if not os.path.exists(args.save_dir): os.mkdir(args.save_dir)
    exp_name = get_formatted_exp_name(args.exp_name, args.resume)
    exp_save_path = os.path.join(args.save_dir, exp_name)
    if not os.path.exists(exp_save_path): os.mkdir(exp_save_path)
    
    with open(os.path.join(exp_save_path, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f)
    
   
    # callback
    checkpoint_callback = ModelCheckpoint(dirpath=exp_save_path, 
                                          save_top_k=args.save_top_k, 
                                          monitor='val_acc',
                                          mode='max',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    
    if not args.resume:
        print("use_formula", args.use_formula)
        model = SpecFuncGroupsClsModel(formula_vocab_size=args.formula_vocab_size,
                                       num_fg_cls=args.num_fg_cls,
                                       use_formula=args.use_formula,
                                       n_spec_f_encoder_layer=args.n_spec_f_encoder_layer,
                                       n_fg_decoder_layer=args.n_fg_decoder_layer,
                                       lr=args.lr,
                                       warm_up_step=args.warmup_steps,
                                       device=device)
    else:
        model = SpecFuncGroupsClsModel.load_from_checkpoint(args.checkpoint)

    if args.code_test:
        wandb_logger = None
        fast_dev_run = 1 # run 1 batch through the trainer to see if there are any bugs
    else:
        wandb_logger = WandbLogger(
            project="IR-FuncGroups-Classify",
            name=exp_name,
            save_dir=exp_save_path
        )
        fast_dev_run = False
    
    if device == "cuda":
        accelerator, devices ='gpu', args.num_gpu
    elif device == "cpu":
        accelerator, devices ='cpu', "auto"
    
    trainer = L.Trainer(accelerator=accelerator,
                        devices=devices,
                        max_epochs=args.n_epochs,
                        fast_dev_run=fast_dev_run, 
                        callbacks=callbacks,
                        logger=wandb_logger,
                        )
    if args.resume:
        trainer.fit(model=model, 
                    ckpt_path=args.diff_checkpoint,
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )
    else:
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )
    trainer.test(model=model, dataloaders=test_loader)
if __name__ == "__main__":
    parser = ArgumentParser()
    # Training
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default='../exp/exp_cls')
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], required=True)
    parser.add_argument("--use_formula", type=bool, default=True)
    parser.add_argument("--formula_vocab_size", type=int, default=26,
                        help='qm9s: 26 | qme14s: 39')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_include_qm9s', action="store_true")
    parser.add_argument("--num_fg_cls", type=int, required=True)
    parser.add_argument("--n_spec_f_encoder_layer", type=int, default=4)
    parser.add_argument("--n_fg_decoder_layer", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lr", type=float, default=0.8)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=17)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code_test", action="store_true")
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--save_top_k", type=int, default=1)
    
    args = parser.parse_args()
    os.environ.pop("SLURM_NTASKS", None)
    torch.set_float32_matmul_precision('medium')
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # debug
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)