import torch

from .dataset.dataset_dataloader import qm9s_dataset_info, qme14s_dataset_info

from .model.diff_with_vae_cls import EnLatentPosDiffusion as EnLatentPosDiffusion_cls

from .model.spec_fg_cls import SpecFuncGroupsClsModel

from .model.vae_ import EGNN_encoder_QM9 as EGNN_encoder_QM9_ff
from .model.vae_ import EGNN_decoder_QM9 as EGNN_decoder_QM9_ff
from .model.vae_ import EnHierarchicaPosVAE as EnHierarchicaPosVAE_cls

from .model.egnn_ import EGNN_dynamics_QM9S
from .model.train_utils import EMACallback

import os
import os.path as osp
import pickle
from pprint import pprint

def load_diffusion(diff_dir_path, device, last_checkpoint=False, checkpoint=None, use_full_cls=False):
    assert osp.exists(diff_dir_path), f"Path do not exist: f{diff_dir_path}"
    with open(osp.join(diff_dir_path, 'args.pickle'), 'rb') as f:
        diff_args = pickle.load(f)
    if not hasattr(diff_args, "use_formula"): diff_args.use_formula=False
    if not hasattr(diff_args, "use_fg"): diff_args.use_fg=False
    print("use_formula", diff_args.use_formula, "use_fg", diff_args.use_fg)
    print("Loading model....")
    if diff_args.ema_decay > 0:
        ema_callback = EMACallback(ema_decay=diff_args.ema_decay)
    else:
        ema_callback = None

    checkpoint = osp.join(diff_dir_path, checkpoint)
    diff_checkpoint_name = osp.basename(checkpoint)
    print(f"Loading: {checkpoint}")
    if use_full_cls:
        diff_model = get_diffusion_model_cls(args=diff_args, device=device, 
                                        ema_callback=ema_callback,
                                        diff_checkpoint=checkpoint,
                                        )
  
    return diff_model, diff_checkpoint_name, diff_args

def get_checkpoint(exp_dir_path, last_checkpoint=False):
    files = os.listdir(exp_dir_path)
    ckpt_name = None
    for f in files:
        if '.ckpt' in f: 
            if (last_checkpoint) and ('last' in f) :
                ckpt_name = f
                break
            elif (not last_checkpoint) and ('last' not in f):
                ckpt_name = f
                break
    if ckpt_name is None:
        return None, None
    else:
        return osp.join(exp_dir_path, ckpt_name), ckpt_name

def get_vae_model_cls(args, device, total_steps, ema_callback):
    if  not hasattr(args, 'dataset'):
        args.dataset = "qm9s"
    if args.dataset == "qm9s":
        num_classes = len(qm9s_dataset_info['atom_decoder'])
    elif args.dataset == "qme14s":
        num_classes = len(qme14s_dataset_info['atom_decoder'])

    if args.h_init_embed:
        in_node_nf = args.dim_zh
    else:
        in_node_nf = num_classes + int(args.include_charges)

    encoder = EGNN_encoder_QM9_ff(in_node_nf=in_node_nf, context_node_nf=0, 
                               out_node_nf=1, n_dims=3, num_classes=num_classes,
                               hidden_nf=args.nf, # hidden_nf: latent dim of h 
                               device=device, act_fn=torch.nn.SiLU(), n_layers=1,
                               attention=args.attention, tanh=args.tanh, norm_constant=args.norm_constant,
                               inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
                               normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
                               include_charges=args.include_charges)
    
    decoder = EGNN_decoder_QM9_ff(in_node_nf=in_node_nf, context_node_nf=0, 
                               out_node_nf=None, n_dims=3, num_classes=num_classes,
                               hidden_nf=args.nf,
                               device=device, act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
                               attention=args.attention, tanh=args.tanh, norm_constant=args.norm_constant,
                               inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
                               normalization_factor=args.normalization_factor, aggregation_method=args.aggregation_method,
                               include_charges=False)

    if not hasattr(args, "use_formula"):
        args.use_formula = False
    
    
    if not hasattr(args, 'use_cross_attn'):
        args.use_cross_attn = True

    assert osp.exists(args.spec_cls_checkpoint)
    print(f"Loading spec_cls_checkpoint: {args.spec_cls_checkpoint}")
    if osp.isdir(args.spec_cls_checkpoint):
        spec_cls_checkpoint, _ = get_checkpoint(args.spec_cls_checkpoint)
    else: spec_cls_checkpoint = args.spec_cls_checkpoint
    if spec_cls_checkpoint is not None:
        print(f"Loading spec_cls_checkpoint: {spec_cls_checkpoint}")
        classifier = SpecFuncGroupsClsModel.load_from_checkpoint(spec_cls_checkpoint)
    else:
        with open(os.path.join(args.spec_cls_checkpoint, 'args.pickle'), 'rb') as f:
            spec_cls_args = pickle.load(f)
        print("spec_cls_args:")
        pprint(vars(spec_cls_args))
        classifier = SpecFuncGroupsClsModel(formula_vocab_size=spec_cls_args.formula_vocab_size,
                                       num_fg_cls=spec_cls_args.num_fg_cls,
                                       use_formula=spec_cls_args.use_formula,
                                       n_spec_f_encoder_layer=spec_cls_args.n_spec_f_encoder_layer,
                                       n_fg_decoder_layer=spec_cls_args.n_fg_decoder_layer,
                                       lr=spec_cls_args.lr,
                                       warm_up_step=spec_cls_args.warmup_steps,
                                       device=device)
        
    vae = EnHierarchicaPosVAE_cls(encoder=encoder,
                            decoder=decoder,
                            spec_cls_model=classifier,
                            cls_loss_weight=args.cls_weight,
                            d_model=args.spec_d_model,
                            spec_len=3200, patch_len=64,
                            in_node_nf=in_node_nf,
                            n_dims=3, num_classes= num_classes,
                            kl_weight=args.kl_weight,
                            h_init_embed=args.h_init_embed,
                            norm_values=args.normalize_factors,
                            include_charges=args.include_charges,
                            augment_noise=args.augment_noise,
                            device=device, 
                            lr=args.lr, warm_up_step=args.warmup_steps, total_steps=total_steps,
                            data_augmentation=args.data_augmentation,
                            ema_callback=ema_callback,
                            use_formula=args.use_formula,
                            use_cross_attn=args.use_cross_attn,
                            )
   
    return vae


def get_diffusion_model_cls(args, device, total_steps=None, ema_callback=None, diff_checkpoint=None):
    """
    Pre-trained VAE: load full spec cls model
    """
    assert osp.exists(args.vae_dir_path), f"Path do not exist: f{args.vae_dir_path}"
    print(f'vae args from: {args.vae_dir_path}')

    with open(osp.join(args.vae_dir_path, 'args.pickle'), 'rb') as f:
        first_stage_args = pickle.load(f)
        print("first_stage_args:")
        pprint(vars(first_stage_args))
    
    vae = get_vae_model_cls(first_stage_args, device, total_steps, ema_callback)

    if diff_checkpoint is None:
        checkpoint, _ = get_checkpoint(args.vae_dir_path)
        if checkpoint is not None:
            print(f"Loading vae checkpoint from {checkpoint}")
            checkpoint = torch.load(checkpoint)
            if "ema_state_dict" in checkpoint:
                vae.load_state_dict(checkpoint["ema_state_dict"])
            else:
                vae.load_state_dict(checkpoint["state_dict"])
    
    print("use_formula: ", args.use_formula, "use_edge: ", args.use_edge)
    if not first_stage_args.h_init_embed:
        in_node_nf = len(qm9s_dataset_info['atom_decoder']) + int(args.include_charges) # dim of z_h
    else:
        in_node_nf = first_stage_args.dim_zh
    print('h_init_embed', first_stage_args.h_init_embed, "in_node_nf", in_node_nf )

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
        print("dynamics_in_node_nf", dynamics_in_node_nf)
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf
    
    if not hasattr(args, "use_edge_cross_attn"):
        args.use_edge_cross_attn = True

  
    net_dynamics = EGNN_dynamics_QM9S(
        d_model=512, spec_len=3200, patch_len=64,
        use_atom_cross_attn=args.use_atom_cross_attn,
        use_edge_cross_attn=args.use_edge_cross_attn,
        use_edge=args.use_edge, in_edge_nf=args.in_edge_nf,
        use_fg=args.use_fg, num_fg_cross_attn_layers=args.num_fg_cross_attn_layers,
        in_node_nf=dynamics_in_node_nf, context_node_nf=0, #args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, 
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, 
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method)
   
    if not hasattr(args, "fix_spec_fg_cls"):
        args.fix_spec_fg_cls = True
    if not hasattr(args, "cls_weight"):
        args.cls_weight = 0

    vdm = EnLatentPosDiffusion_cls(
        vae=vae,
        dynamics=net_dynamics,
        in_node_nf=in_node_nf,
        n_dims=3,
        use_formula=args.use_formula, use_fg=args.use_fg, 
        fix_spec_fg_cls=args.fix_spec_fg_cls, cls_weight=args.cls_weight,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges,
        trainable_ae=args.trainable_ae,
        augment_noise=args.augment_noise,
        device=device, 
        lr=args.lr, warm_up_step=args.warmup_steps, total_steps=total_steps,
        data_augmentation=args.data_augmentation,
        ema_callback=ema_callback
        )
  
    if diff_checkpoint is not None:
        print("Loading diffusion checkpoint from: ", diff_checkpoint)
        checkpoint = torch.load(diff_checkpoint)

        new_state_dict = {}
        for name, param in checkpoint['ema_state_dict'].items():
            if "edge_cross_attn.cross_attn." in name:
                name = name.split(".")
                new_name = name[:2] + ['cross_attn_with_spec'] + name[3:]
                new_name = ".".join(new_name)
                new_state_dict[new_name] = param
            elif "cross_attn_with_fg" in name:
                new_state_dict[name] = param
            else:
                new_state_dict[name] = param
        vdm.load_state_dict(new_state_dict)
    return vdm
