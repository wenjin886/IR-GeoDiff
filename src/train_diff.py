import torch
from torch.utils.data import DataLoader



from .code.dataset.dataset_dataloader import (prep_ir_geo_dataset, 
                                         qm9s_dataset_collate_fn,
                                         qme14s_dataset_collate_fn,
                                         )
from .code.get_model import get_diffusion_model_cls
from .code.model.train_utils import EMACallback
                             
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger


from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime
import pickle
import os


def get_formatted_exp_name(exp_name, resume=False):
    formatted_time = datetime.now().strftime("%H-%M-%d-%m-%Y")
    if resume and ("resume" not in exp_name):
        formatted_exp_name = f"resume_{exp_name}_{formatted_time}"
    else:
        formatted_exp_name = f"{exp_name}_{formatted_time}"
    return formatted_exp_name





def main(args):
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    if torch.cuda.is_available():
        print(f"Current CUDA seed: {torch.cuda.initial_seed()}")
    torch.set_float32_matmul_precision(args.precision)

    if args.resume:
        assert args.diff_checkpoint is not None and os.path.exists(args.diff_checkpoint), f"Please check diff_checkoint: {args.diff_checkpoint}."
    


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
        train_loader = DataLoader(
                                #   Subset(dataset["train"], list(range(100))), 
                                  dataset["train"], 
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
    
    
    if not args.code_test:
        if (not os.path.exists(args.save_dir)): os.mkdir(args.save_dir)
        exp_name = get_formatted_exp_name(args.exp_name, args.resume)
        exp_save_path = os.path.join(args.save_dir, exp_name)
        if (not os.path.exists(exp_save_path)): os.mkdir(exp_save_path)
        
        with open(os.path.join(exp_save_path, 'args.pickle'), 'wb') as f:
            pickle.dump(args, f)
    else:
        exp_save_path = "./"
        
    # cal total_step
    batches_per_epoch = len(train_loader)
    total_steps = int(args.n_epochs * batches_per_epoch)
    print("total_steps", total_steps)
    print("lr", args.lr, type(args.lr), "warmup_steps", args.warmup_steps)
    
    # callback
    best_dir_path = os.path.join(exp_save_path, "best")
    os.makedirs(best_dir_path, exist_ok=True)
    best_checkpoint_callback = ModelCheckpoint(dirpath=best_dir_path,
                                          save_top_k=args.save_top_k, 
                                          monitor=args.loss_monitor,
                                        #   monitor='val_loss_recon',
                                          save_last=True)

    callbacks = [best_checkpoint_callback]                                    
    if args.save_every_n_epochs is not None:
        periodic_dir_path = os.path.join(exp_save_path, "periodic")
        os.makedirs(periodic_dir_path, exist_ok=True)
        periodic_checkpoint_callback = ModelCheckpoint(
                                        dirpath=periodic_dir_path,
                                        every_n_epochs=args.save_every_n_epochs,
                                        save_top_k=-1  # 保存所有
                                    )
        callbacks.append(periodic_checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    if args.ema_decay > 0:
        ema_callback = EMACallback(ema_decay=args.ema_decay)
        callbacks.append(ema_callback)
    else:
        ema_callback = None
        
    print("callbacks", len(callbacks))
    
    # if args.fix_spec_fg_cls or args.cls_weight > 0:
    # if args.use_vae_spec_cls:
        # print("get_diffusion_model_cls")
    model = get_diffusion_model_cls(args, device, total_steps, ema_callback)
    # else:
        # model = get_diffusion_model_ff(args, device, total_steps, ema_callback)

    if args.code_test:
        wandb_logger = None
        fast_dev_run = 1 # run 1 batch through the trainer to see if there are any bugs
    else:
        wandb_logger = WandbLogger(
            # project="IR-Geo-Latent-Diff",
            project=args.wandb_project,
            name=exp_name,
            save_dir=exp_save_path
        )
        fast_dev_run = False
    
    if args.num_gpu == 1: strategy="auto"
    else: 
        strategy = "ddp"

    print("gpu num: ", args.num_gpu, "strategy: ",strategy)

    trainer = L.Trainer(accelerator='gpu',
                        devices=args.num_gpu,
                        strategy=strategy,
                        max_epochs=args.n_epochs,
                        fast_dev_run=fast_dev_run, 
                        callbacks=callbacks,
                        logger=wandb_logger,
                        gradient_clip_val=0, 
                        gradient_clip_algorithm=None,
                        accumulate_grad_batches=args.accumulate_grad_batches
                        )
    if args.resume:
        trainer.fit(model=model, 
                    ckpt_path=args.diff_checkpoint,
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )
        
    else:
        if (args.diff_checkpoint is not None) : # load para but train from the begining
            checkpoint = torch.load(args.diff_checkpoint)
            if not args.train_qme14s_from_qm9s:
                model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
            
            else: # train_qme14s_from_qm9s
                # exclude_names = ['vae.spec_cls_model.fg_queries', 'vae.spec_cls_model.formula_embed.0.lut.weight', 'vae.h_embed.lut.weight']
                exclude_names = 'vae' # use pretrained vae model for qme14s
                print("Loading pretrained QM9S model...")
                # exclude_names = ['vae.spec_cls_model', 'vae.h_embed.lut.weight']
                print("exclude_names", exclude_names)
                new_state_dict = {}
                for name, param in model.state_dict().items():
                    if type(exclude_names) == list:
                        if (name in checkpoint['ema_state_dict']) and (name not in exclude_names): # use dm model for qm9s
                            new_state_dict[name] = checkpoint['ema_state_dict'][name]
                        else:
                            new_state_dict[name] = param
                    elif type(exclude_names) == str:
                        if exclude_names in name: # use pretrained vae model for qme14s
                            print("Excluding", name)
                            new_state_dict[name] = param
                        else: # use dm model for qm9s
                            new_state_dict[name] = checkpoint['ema_state_dict'][name]
                    
                
                model.load_state_dict(new_state_dict, strict=True)
                # raise ValueError('stop here')

        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=val_loader,
                    )

    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='diffusion')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument("--dataset", choices=["qm9s", "qme14s"], required=True)
    parser.add_argument("--num_fg_cls", type=int)
    parser.add_argument('--use_edge', action='store_true')
    parser.add_argument('--in_edge_nf', type=int, default=1)
    parser.add_argument('--code_test', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='IR-Geo-Latent-Diff')
    parser.add_argument('--save_dir', type=str, default='../exp/exp_diff')
    parser.add_argument('--loss_monitor', type=str, default='val_loss')
    parser.add_argument('--vae_dir_path', type=Path, default=None,
                        help='pre-trained vae dir containing checkpoint and args')
    parser.add_argument('--trainable_ae', type=eval, default=True)
    parser.add_argument('--num_gpu', type=int, default=1)

    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--diff_checkpoint", type=Path, default=None)
    parser.add_argument('--train_qme14s_from_qm9s', action='store_true')

    parser.add_argument("--use_atom_cross_attn", type=eval, default=True)
    parser.add_argument("--use_edge_cross_attn", type=eval, default=True)
    parser.add_argument("--use_formula", action="store_true")
    parser.add_argument("--use_fg", action="store_true")
    parser.add_argument("--num_fg_cross_attn_layers", type=int, default=2)
    parser.add_argument("--spec_fg_cls_checkpoint", type=str)
 

    # parser.add_argument('--CA_modified', action='store_true', 
                        # help="modify egnn to satisfy equivariant and invariant, for rebuttal")

    # diff_with_vae_cls
    # parser.add_argument('--use_vae_spec_cls', action='store_true')
    parser.add_argument('--fix_spec_fg_cls', action='store_true')
    parser.add_argument('--cls_weight', type=float, default=0)


    # Diffusion
    parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                        help='learned, cosine')
    parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5)
    parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                        choices=['vlb', 'l2'])
    
    # VAE
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
    parser.add_argument('--latent_nf', type=int, default=1,
                        help='number of latent features')
    parser.add_argument('--nf', type=int, default=256,
                        help='latent features num of h')
    parser.add_argument('--n_layers', type=int, default=9,
                        help='number of layers')
    parser.add_argument('--kl_weight', type=float, default=0.01,
                        help='weight of KL term in ELBO')
    parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 10],
                    help='normalize factors for [x, categorical, integer]')
    
    # Train
    parser.add_argument('--cuda', type=eval, default=True)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_top_k', type=int, default=1,
                        help="Number of saved checkpoints")
    parser.add_argument('--save_every_n_epochs', type=eval, default=None)
    parser.add_argument('--num_workers', type=int, default=17)
    parser.add_argument('--lr', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=3000)
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    parser.add_argument('--augment_noise', type=float, default=0)
    parser.add_argument('--data_augmentation', type=eval, default=False)
    parser.add_argument('--precision', type=str, default='medium',choices=['medium', 'high'])
    # parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    args = parser.parse_args()

    os.environ.pop("SLURM_NTASKS", None)
    main(args)
    
