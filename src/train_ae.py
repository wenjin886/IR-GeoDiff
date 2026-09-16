import torch
from torch.utils.data import DataLoader

from .code.dataset.dataset_dataloader import (prep_ir_geo_dataset, 
                                         qm9s_dataset_collate_fn,
                                         qme14s_dataset_collate_fn,
                                         )

from .code.get_model import get_vae_model_cls
from .code.model.train_utils import EMACallback

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import os
import pickle
import copy

from pprint import pprint

def get_formatted_exp_name(exp_name):
    formatted_time = datetime.now().strftime("%H-%M-%m-%d-%Y")
    return f"{exp_name}_{formatted_time}"

def main(args):
    pprint(vars(args))
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    if torch.cuda.is_available():
        print(f"Current CUDA seed: {torch.cuda.initial_seed()}")
    torch.set_float32_matmul_precision(args.precision)


    dataset = prep_ir_geo_dataset(args.data_dir, args.dataset)

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
    exp_name = get_formatted_exp_name(args.exp_name)
    exp_save_path = os.path.join(args.save_dir, exp_name)
    if not os.path.exists(exp_save_path): os.mkdir(exp_save_path)
    with open(os.path.join(exp_save_path, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f)
    
    if args.code_test:
        wandb_logger = None
        fast_dev_run = 1 # runs 1 batch through the trainer to see if there are any bugs
    else:
        wandb_logger = WandbLogger(
            project="IR-Geo-Latent-Diff",
            name=exp_name,
            save_dir=exp_save_path
        )
        fast_dev_run = False

    checkpoint_callback = ModelCheckpoint(dirpath=exp_save_path, 
                                          save_top_k=args.save_top_k, 
                                          monitor='val_loss',
                                          save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    if args.ema_decay > 0:
        ema_callback = EMACallback(ema_decay=args.ema_decay)
        callbacks = [checkpoint_callback, lr_monitor, ema_callback]
    else:
        ema_callback = None
        callbacks = [checkpoint_callback, lr_monitor]
    
    # cal total_step
    batches_per_epoch = len(train_loader)
    total_steps = int(args.n_epochs * batches_per_epoch)
    # if hasattr(args, "use_spec_cls") and args.use_spec_cls:
    model = get_vae_model_cls(args, device, total_steps, ema_callback)
    # else:
        # model = get_vae_model_ff(args, device, total_steps, ema_callback)
    

    trainer = L.Trainer(accelerator='gpu',
                        devices=1,
                        max_epochs=args.n_epochs,
                        fast_dev_run=fast_dev_run, 
                        callbacks=callbacks,
                        logger=wandb_logger,
                        gradient_clip_val=0, 
                        gradient_clip_algorithm=None,
                        accumulate_grad_batches=args.accumulate_grad_batches
                        )
    if args.resume:
        print(f"Resuming from checkpoint: {args.vae_checkpoint}")
        trainer.fit(model=model, 
                    ckpt_path=args.vae_checkpoint,
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )
    else:
        if (args.vae_checkpoint is not None) : # load para but train from the begining
            checkpoint = torch.load(args.vae_checkpoint)
            if not args.train_qme14s_from_qm9s:
                model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
            
            else:
                exclude_names = [
                    'spec_cls_model.fg_queries', 
                    'spec_cls_model.formula_embed.0.lut.weight',
                    'h_embed.lut.weight']
                new_state_dict = {}
                for name, param in model.state_dict().items():
                    if (name in checkpoint['ema_state_dict']) and (name not in exclude_names): 
                        new_state_dict[name] = checkpoint['ema_state_dict'][name]
                    else:
                        new_state_dict[name] = param
                print("Loading pretrained QM9S model...")
                model.load_state_dict(new_state_dict, strict=True)

        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )
    
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='vae')
    parser.add_argument('--save_dir', type=str, default='../exp/exp_vae')
    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--h_init_embed', type=eval, default=True)
    parser.add_argument('--dim_zh', type=int, default=16)
    parser.add_argument('--use_formula', type=eval, default=True)
    parser.add_argument("--use_cross_attn", type=eval, default=True)
    parser.add_argument('--spec_cls_checkpoint', type=str, default="None")
    # parser.add_argument('--spec_cls_weight', type=float, default=0)
    parser.add_argument('--use_spec_cls', type=eval, default=True)
    parser.add_argument("--fix_cls_model", action='store_true')
    # parser.add_argument('--ctr_weight', type=float, default=0,
                        # help='weight of contrastive loss')
    # parser.add_argument('--qm9s_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], required=True)
    parser.add_argument('--split_file', type=str, default='')
    parser.add_argument('--use_fg', action='store_true')
    parser.add_argument('--cls_weight',type=float, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--vae_checkpoint", type=Path, default=None)
    parser.add_argument('--train_qme14s_from_qm9s', action='store_true')

    # Model
    parser.add_argument('--spec_d_model', type=int, default=512)
    parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
    parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
    parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
    parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
    parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
    parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
    parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
    parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
    # parser.add_argument('--latent_nf', type=int, default=1,
    #                     help='dim of encoder output of h')
    parser.add_argument('--nf', type=int, default=256,
                        help='dim of h in vae encoder and decoder')
    parser.add_argument('--n_layers', type=int, default=9,
                        help='number of layers')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='weight of KL term in ELBO')
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
    
    # Train
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_top_k', type=int, default=1,
                        help="Number of saved checkpoints")
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--data_augmentation', type=eval, default=False)
    parser.add_argument('--num_workers', type=int, default=17)
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_gpu', type=int, default=1)
    args = parser.parse_args()

    os.environ.pop("SLURM_NTASKS", None)
    # print(args)
    main(args)
    
