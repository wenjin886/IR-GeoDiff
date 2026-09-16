import torch
from torch import nn, optim
# from torch.nn import functional as F

import lightning as L

from .model_utils import (remove_mean, remove_mean_with_mask,
                        #   sample_gaussian_with_mask,
                          sample_center_gravity_zero_gaussian_with_mask,
                          assert_correctly_masked,
                          assert_mean_zero_with_mask)
from .train_utils import (random_rotation, check_mask_correct,
                          Queue, gradient_clipping)
from .loss_utils import (sum_except_batch,
                        #  gaussian_KL, 
                         gaussian_KL_for_dimension)
from .egnn_ import EGNN
from .SpecEncoder_with_formula_fg import SpecEncoder, SpecAtomCrossAttention
from .spec_utils import Embeddings
from .spec_fg_cls import SpecFuncGroupsClsModel



class EGNN_encoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, num_classes, 
                 hidden_nf=256, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        '''
        :param in_node_nf: Number of invariant features for input nodes.'''
        super().__init__()

        include_charges = int(include_charges)

        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, out_node_nf=hidden_nf, 
            in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf
      
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, out_node_nf * 2 + 1))

        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device_ = device
        self.n_dims = n_dims
        self._edges_dict = {}

        self.out_node_nf = out_node_nf

    # def forward(self, t, xh, node_mask, edge_mask, context=None):
    #     raise NotImplementedError

    # def wrap_forward(self, node_mask, edge_mask, context):
    #     def fwd(time, state):
    #         return self._forward(time, state, node_mask, edge_mask, context)
    #     return fwd

    # def unwrap_forward(self):
    #     return self._forward

    def forward(self, xh, node_mask, edge_mask, context):      
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device_) # full-connected graph with n nodes, list: [len=2, item:(tensor, shape:(n**2,))]
        edges = [x.to(self.device_) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device_)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = x_final * node_mask  # This masking operation is redundant but just in case
        
        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        h_final = self.final_mlp(h_final)
        h_final = h_final * node_mask if node_mask is not None else h_final
        h_final = h_final.view(bs, n_nodes, -1)

        vel_mean = vel
        vel_std = h_final[:, :, :1].sum(dim=1, keepdim=True).expand(-1, n_nodes, -1)
        vel_std = torch.exp(0.5 * vel_std)

        h_mean = h_final[:, :, 1:1 + self.out_node_nf]
        h_std = torch.exp(0.5 * h_final[:, :, 1 + self.out_node_nf:])

        if torch.any(torch.isnan(vel_std)):
            print('Warning: detected nan in vel_std, resetting to one.')
            vel_std = torch.ones_like(vel_std)
        
        if torch.any(torch.isnan(h_std)):
            print('Warning: detected nan in h_std, resetting to one.')
            h_std = torch.ones_like(h_std)
        
        # Note: only vel_mean and h_mean are correctly masked
        # vel_std and h_std are not masked, but that's fine:

        # For calculating KL: vel_std will be squeezed to 1D
        # h_std will be masked

        # For sampling: both stds will be masked in reparameterization

        return vel_mean, vel_std, h_mean, h_std
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

class EGNN_decoder_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf, out_node_nf,
                 n_dims, num_classes,
                 hidden_nf=256, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 tanh=False, norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 include_charges=True):
        super().__init__()

        include_charges = int(include_charges)
        if out_node_nf is None:
            out_node_nf = in_node_nf
        # num_classes = out_node_nf - include_charges
        num_classes = num_classes

        self.egnn = EGNN(
            in_node_nf=in_node_nf + context_node_nf, out_node_nf=out_node_nf, 
            in_edge_nf=1, hidden_nf=hidden_nf, device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
            inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method)
        self.in_node_nf = in_node_nf
     
        self.num_classes = num_classes
        self.include_charges = include_charges
        self.context_node_nf = context_node_nf
        self.device_ = device
        self.n_dims = n_dims
        self._edges_dict = {}


    def forward(self, xh, node_mask, edge_mask, context=None):
        # print("vae decoder xh", xh.shape)
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device_)
        edges = [x.to(self.device_) for x in edges]
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        xh = xh.view(bs*n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device_)
        else:
            h = xh[:, self.n_dims:].clone()

        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)


        h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
        vel = x_final * node_mask  # This masking operation is redundant but just in case

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if node_mask is not None:
            h_final = h_final * node_mask
        h_final = h_final.view(bs, n_nodes, -1)

        return vel, h_final
    
    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

class EnHierarchicaPosVAE(L.LightningModule):
    def __init__(self, 
                 use_formula: bool,
                 encoder: EGNN_encoder_QM9, 
                 decoder: EGNN_decoder_QM9,
                 d_model, spec_len, patch_len,
                 in_node_nf: int, n_dims: int, num_classes: int,
                 kl_weight: float,
                 spec_cls_model: SpecFuncGroupsClsModel=None,
                 cls_loss_weight=0,
                 fix_cls_model=False,
                 h_init_embed=False,
                 use_cross_attn=True,
                 norm_values=(1., 1., 1.), 
                 norm_biases=(None, 0., 0.), 
                 include_charges=True,
                 device='cuda',
                 augment_noise=0,
                 lr=2e-4, warm_up_step=6000, total_steps=None,
                 data_augmentation=False,
                 ema_callback=None):
        """
        args:
            in_node_nf: dims of h; if not h_init_embed: in_node_nf should be the sum of num_classes and include_charges (0 or 1)
            n_dims: dims of positions, usually 3
            num_classes: num of atom types
        """
        super().__init__()

        
        self.encoder = encoder
        self.decoder = decoder
        

        self.include_charges = include_charges

        self.in_node_nf = in_node_nf 
        self.n_dims = n_dims 
        self.num_classes = num_classes
        self.kl_weight = kl_weight

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))
        
        self.device_ = torch.device(device)
        self.dtype_ = torch.float32
        self.augment_noise = augment_noise
        self.lr = lr
        self.warm_up_step = warm_up_step
        self.total_steps = total_steps
        self.data_augmentation = data_augmentation

        if spec_cls_model is not None:
            self.use_spec_cls_model = True
            self.spec_cls_model = spec_cls_model
            self.cls_loss_weight = cls_loss_weight
            
            if fix_cls_model:
                assert cls_loss_weight == 0
                for param in self.spec_cls_model.parameters():
                    param.requires_grad = False            

        else:
            self.use_spec_cls_model = False
            self.spec_embed = SpecEncoder(device=self.device_, d_model=d_model,
                                        spec_len=spec_len, 
                                        patch_len=patch_len,
                                        use_formula=use_formula
                                        )
        
        self.use_formula = use_formula
        print("vae | use_formula", use_formula)
        
        self.h_init_embed = h_init_embed
        print("vae | in_node_nf", self.in_node_nf, "| num_classes", self.num_classes)
        if self.h_init_embed:
            # H, C, N, O, F
            self.h_embed = Embeddings(d_model=self.in_node_nf, vocab=self.num_classes)

        self.use_cross_attn = use_cross_attn
        print("vae | use_cross_attn", self.use_cross_attn)
        if self.use_cross_attn:
            self.cross_attn = SpecAtomCrossAttention(device=self.device_, 
                                                    d_atom=self.in_node_nf, 
                                                    d_model=d_model,
                                                    spec_patch_num=int(spec_len/patch_len))

        # clip grad
        self.gradnorm_queue = Queue(max_len=50)
        self.gradnorm_queue.add(3000)  

        # EMA (Exponential Moving Average)
        self.ema_callback = ema_callback

        self.save_hyperparameters(ignore=['encoder','decoder', 'spec_cls_model'])

    def get_z_h_spec_embed(self, spec, h, node_mask, formula_ids=None):
        """
        Returns:
            z_h: latent node features
            spec: encoded
        """
        # Encode spectra

        spec_att_mask = None
        # print('vae use formula', self.use_formula)
        if self.use_spec_cls_model:
            spec, spec_att_mask = self.spec_cls_model.encode(formula_ids, spec)
        else:
            if self.use_formula:
                spec, spec_att_mask = self.spec_embed(spec, formula_ids)
            else:
                spec, _ = self.spec_embed(spec)
        
        # Obtain z_h
        if self.h_init_embed:
            h = torch.argmax(h['categorical'], dim=-1)
            h = self.h_embed(h)
        else:
            h = torch.cat([h['categorical'], h['integer']], dim=2) # (bs, N, in_node_nf)
        
        if self.use_cross_attn:
            z_h = self.cross_attn(h, node_mask, spec, spec_att_mask)
        else:
            z_h = h
        return z_h, spec, spec_att_mask

    def encode(self, x, z_h, node_mask=None, edge_mask=None, context=None):
        """Computes q(z|x)."""
        
        x_zh = torch.cat([x, z_h], dim=2)

        assert_mean_zero_with_mask(x_zh[:, :, :self.n_dims], node_mask)

        # Encoder output.
        z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = self.encoder(x_zh, node_mask, edge_mask, context)

        bs, _, _ = z_x_mu.size()
        sigma_0_x = torch.ones(bs, 1, 1).to(z_x_mu) * 0.0032

        return z_x_mu, sigma_0_x
    
    def decode(self, z_x, z_h, node_mask=None, edge_mask=None, context=None):
        """Computes p(x|z)."""
        
        # Decoder output (reconstruction).
        zx_zh = torch.cat([z_x, z_h], dim=2)
        x_recon, _ = self.decoder(zx_zh, node_mask, edge_mask, context)
        assert_mean_zero_with_mask(x_recon, node_mask)

        x = x_recon
        assert_correctly_masked(x, node_mask)
        return x
    
    
    def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x
    
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    
    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims
    
    # def compute_reconstruction_error(self, xh_rec, xh):
    def compute_reconstruction_error(self, x_rec, x):
        """Computes reconstruction error."""
        
        error_x = sum_except_batch((x_rec - x) ** 2)
        error = error_x #+ error_h_cat + error_h_int

        if self.training:
            denom = self.n_dims * x.shape[1]
            error = error / denom

        return error
    
    
    def compute_loss(self, x, z_h, node_mask, edge_mask, context, spec, spec_att_mask, fg_onehot):
        """Computes an estimator for the variational lower bound."""

        # Encoder output.
        z_x_mu, z_x_sigma = self.encode(x, z_h, node_mask, edge_mask, context)
        

        # KL for equivariant features.
        assert z_x_sigma.mean(dim=(1,2), keepdim=True).expand_as(z_x_sigma).allclose(z_x_sigma, atol=1e-7)
        zeros, ones = torch.zeros_like(z_x_mu), torch.ones_like(z_x_sigma.mean(dim=(1,2)))
        subspace_d = self.subspace_dimensionality(node_mask)
        loss_kl_x = gaussian_KL_for_dimension(z_x_mu, ones, zeros, ones, subspace_d)
        loss_kl = loss_kl_x

        # Infer latent z.
        z_x = self.sample_normal(z_x_mu, z_x_sigma.expand(-1, -1, 3), node_mask)
        assert z_x.shape[2] == self.n_dims
        assert_correctly_masked(z_x, node_mask)
        assert_mean_zero_with_mask(z_x, node_mask)

        zx_zh = torch.cat([z_x, z_h], dim=2)
        x_recon, _ = self.decoder(zx_zh, node_mask, edge_mask, context)
        loss_recon = self.compute_reconstruction_error(x_recon, x)

        if self.cls_loss_weight > 0:
            # Compute the classification loss
            fg_queries_ = self.spec_cls_model.decode(self.spec_cls_model.fg_queries, spec, spec_att_mask)
            logits = self.spec_cls_model.cls_head(fg_queries_).squeeze(-1)
            cls_loss = self.spec_cls_model._cal_loss(logits, fg_onehot)
            
            # Combining the terms
            assert loss_recon.size() == loss_kl.size()
            loss = loss_recon + self.kl_weight * loss_kl + self.cls_loss_weight * cls_loss

            assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

            return loss, {'loss_kl': loss_kl.squeeze().detach().mean(), 'rec_error': loss_recon.squeeze().detach().mean(), "cls_loss": cls_loss.squeeze().detach().mean()}
        else:
            # Combining the terms
            assert loss_recon.size() == loss_kl.size()
            loss = loss_recon + self.kl_weight * loss_kl

            assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

            return loss, {'loss_kl': loss_kl.squeeze().detach().mean(), 'rec_error': loss_recon.squeeze().detach().mean()}
    
    def forward(self, spec, x, h, node_mask=None, edge_mask=None, context=None, 
                formula_ids=None, fg_onehot=None):
        """
        Computes the ELBO if training. And if eval then always computes NLL.
        """

        if self.use_formula:
            assert formula_ids is not None, 'Please check the input of formula.'
            z_h, spec, spec_att_mask = self.get_z_h_spec_embed(spec, h, node_mask, formula_ids)
        else:
            z_h, spec, spec_att_mask = self.get_z_h_spec_embed(spec, h, node_mask)
        
        loss, loss_dict = self.compute_loss(x, z_h, node_mask, edge_mask, context, 
                                            spec, spec_att_mask, fg_onehot)

        return loss, loss_dict

    
    def training_step(self, batch, batch_idx):
        #  training_step defines the train loop.
        # it is independent of forward
        x = batch['positions'].to(self.device_, self.dtype_)
        bs = x.shape[0]
        node_mask = batch['atom_mask'].to(self.device_, self.dtype_).unsqueeze(2)
        edge_mask = batch['edge_mask'].to(self.device_, self.dtype_)
        one_hot = batch['one_hot'].to(self.device_, self.dtype_)
        charges = (batch['charges'] if self.include_charges else torch.zeros(0)).to(self.device_, self.dtype_)
        spec = batch['spectra'].to(self.device_, self.dtype_)
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        if self.use_formula:
            formula_ids = batch["formula"].to(self.device_).int()
        else: formula_ids = None
        
        x = remove_mean_with_mask(x, node_mask)

        if self.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.augment_noise

        x = remove_mean_with_mask(x, node_mask)

        if self.data_augmentation:
            x = random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}
        

        # transform batch through flow
        loss, loss_dict  = self(spec, x, h, node_mask, edge_mask, context=None, formula_ids=formula_ids, fg_onehot=fg_onehot)
        loss = loss.mean()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_loss_kl", loss_dict["loss_kl"], batch_size=bs)
        self.log("train_loss_recon", loss_dict["rec_error"], batch_size=bs)
        if self.cls_loss_weight > 0:
            self.log("train_cls_loss", loss_dict["cls_loss"], batch_size=bs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #  training_step defines the train loop.
        # it is independent of forward
        x = batch['positions'].to(self.device_, self.dtype_)
        bs = x.shape[0]
        node_mask = batch['atom_mask'].to(self.device_, self.dtype_).unsqueeze(2)
        edge_mask = batch['edge_mask'].to(self.device_, self.dtype_)
        one_hot = batch['one_hot'].to(self.device_, self.dtype_)
        charges = (batch['charges'] if self.include_charges else torch.zeros(0)).to(self.device_, self.dtype_)
        spec = batch['spectra'].to(self.device_, self.dtype_)
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        if self.use_formula:
            formula_ids = batch["formula"].to(self.device_).int()
        else: formula_ids = None

        if self.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

      

        # transform batch through flow
        loss, loss_dict = self(spec, x, h, node_mask, edge_mask, context=None, formula_ids=formula_ids, fg_onehot=fg_onehot)
        loss = loss.mean()
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, batch_size=bs)
        self.log("val_loss_kl", loss_dict["loss_kl"], batch_size=bs)
        self.log("val_loss_recon", loss_dict["rec_error"], batch_size=bs)
        if self.cls_loss_weight > 0:
            self.log("val_cls_loss", loss_dict["cls_loss"], batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        #  training_step defines the train loop.
        # it is independent of forward
        x = batch['positions'].to(self.device_, self.dtype_)
        bs = x.shape[0]
        node_mask = batch['atom_mask'].to(self.device_, self.dtype_).unsqueeze(2)
        edge_mask = batch['edge_mask'].to(self.device_, self.dtype_)
        one_hot = batch['one_hot'].to(self.device_, self.dtype_)
        charges = (batch['charges'] if self.include_charges else torch.zeros(0)).to(self.device_, self.dtype_)
        spec = batch['spectra'].to(self.device_, self.dtype_)
        fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        if self.use_formula:
            formula_ids = batch["formula"].to(self.device_).int()
        else: formula_ids = None

        if self.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * self.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

     

        # transform batch through flow
        loss, loss_dict = self(spec, x, h, node_mask, edge_mask, context=None, formula_ids=formula_ids, fg_onehot=fg_onehot)
        loss = loss.mean()
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, batch_size=bs)
        self.log("test_loss_kl", loss_dict["loss_kl"], batch_size=bs)
        self.log("test_loss_recon", loss_dict["rec_error"], batch_size=bs)
        if self.cls_loss_weight > 0:
            self.log("test_cls_loss", loss_dict["cls_loss"], batch_size=bs)

        return loss
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.lr,
                                amsgrad=True,
                                weight_decay=1e-12)

        def rate(step):
           
            if step == 0:
                step = 1
            lr_scale = 1 * (
                512 ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            ) 

            return lr_scale

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        gradient_clipping(self, self.gradnorm_queue)
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.state_dict()
        if self.ema_callback is not None:
            checkpoint['ema_state_dict'] = self.ema_callback.ema_model.state_dict()



