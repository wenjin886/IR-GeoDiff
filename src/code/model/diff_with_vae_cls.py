import torch
from torch import nn, optim
from torch.nn import functional as F

import lightning as L

import numpy as np
import math

from .model_utils import (remove_mean_with_mask,
                          sample_center_gravity_zero_gaussian_with_mask,
                          assert_correctly_masked,
                          assert_mean_zero_with_mask)

from .loss_utils import (sum_except_batch,
                         gaussian_KL, 
                         gaussian_KL_for_dimension)

from .train_utils import (random_rotation, check_mask_correct,
                          Queue, gradient_clipping,
                          )

from .egnn_ import EGNN_dynamics_QM9S
from .vae_ import EnHierarchicaPosVAE


class PositiveLinear(nn.Module):
    """Linear layer with weights forced to be positive."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 weight_init_offset: int = -2):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init_offset = weight_init_offset
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        with torch.no_grad():
            self.weight.add_(self.weight_init_offset)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        positive_weight = F.softplus(self.weight)
        return F.linear(input, positive_weight, self.bias)

class GammaNetwork(nn.Module):
    """The gamma network models a monotonic increasing function. Construction as in the VDM paper."""
    def __init__(self):
        super().__init__()

        self.l1 = PositiveLinear(1, 1)
        self.l2 = PositiveLinear(1, 1024)
        self.l3 = PositiveLinear(1024, 1)

        self.gamma_0 = nn.Parameter(torch.tensor([-5.]))
        self.gamma_1 = nn.Parameter(torch.tensor([10.]))
        self.show_schedule()

    def show_schedule(self, num_steps=50):
        t = torch.linspace(0, 1, num_steps).view(num_steps, 1)
        gamma = self.forward(t)
        print('Gamma schedule:')
        print(gamma.detach().cpu().numpy().reshape(num_steps))

    def gamma_tilde(self, t):
        l1_t = self.l1(t)
        return l1_t + self.l3(torch.sigmoid(self.l2(l1_t)))

    def forward(self, t):
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        # Not super efficient.
        gamma_tilde_0 = self.gamma_tilde(zeros)
        gamma_tilde_1 = self.gamma_tilde(ones)
        gamma_tilde_t = self.gamma_tilde(t)

        # Normalize to [0, 1]
        normalized_gamma = (gamma_tilde_t - gamma_tilde_0) / (
                gamma_tilde_1 - gamma_tilde_0)

        # Rescale to [gamma_0, gamma_1]
        gamma = self.gamma_0 + (self.gamma_1 - self.gamma_0) * normalized_gamma

        return gamma

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power))**2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod

def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """
    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)
        elif 'polynomial' in noise_schedule:
            splits = noise_schedule.split('_')
            assert len(splits) == 2
            power = float(splits[1])
            alphas2 = polynomial_schedule(timesteps, s=precision, power=power)
        else:
            raise ValueError(noise_schedule)

        print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class EnLatentPosDiffusion(L.LightningModule):
    """
    The E(n) Diffusion Module which only adds noise on positions.
    """
    def __init__(
            self,
            vae: EnHierarchicaPosVAE,
            dynamics: EGNN_dynamics_QM9S, 
            in_node_nf: int, n_dims: int,
            use_formula: bool=False, use_fg: bool=False, # cross-attention
            fix_spec_fg_cls: bool=True, cls_weight: float=0, # spec_fg_cls=None, 
            timesteps: int = 1000, parametrization='eps', noise_schedule='learned',
            noise_precision=1e-4, loss_type='l2', norm_values=(1., 1., 1.),
            norm_biases=(None, 0., 0.), 
            include_charges=True,
            trainable_ae=True,
            device='cuda',
            augment_noise=0,
            lr=2e-4, warm_up_step=6000, total_steps=None,
            data_augmentation=False,
            ema_callback=None):
        super().__init__()
        
        self.trainable_ae = trainable_ae
        self.instantiate_first_stage(vae=vae)

        assert loss_type in {'vlb', 'l2'}
        self.loss_type = loss_type
        self.include_charges = include_charges
        if noise_schedule == 'learned':
            assert loss_type == 'vlb', 'A noise schedule can only be learned' \
                                       ' with a vlb objective.'

        # Only supported parametrization.
        assert parametrization == 'eps'

        if noise_schedule == 'learned':
            self.gamma = GammaNetwork()
        else:
            self.gamma = PredefinedNoiseSchedule(noise_schedule, timesteps=timesteps,
                                                 precision=noise_precision)

        # The network that will predict the denoising.
        self.dynamics = dynamics

        self.in_node_nf = in_node_nf
        self.n_dims = n_dims
        self.num_classes = self.in_node_nf - self.include_charges

        self.T = timesteps
        self.parametrization = parametrization

        self.norm_values = norm_values
        self.norm_biases = norm_biases
        self.register_buffer('buffer', torch.zeros(1))

        if noise_schedule != 'learned':
            self.check_issues_norm_values()
        
        self.device_ = torch.device(device)
        self.dtype_ = torch.float32
        self.augment_noise = augment_noise
        self.lr = lr
        self.warm_up_step = warm_up_step
        self.total_steps = total_steps
        self.data_augmentation = data_augmentation

        # clip grad
        self.gradnorm_queue = Queue(max_len=50)
        self.gradnorm_queue.add(3000) 
        

        # EMA (Exponential Moving Average)
        self.ema_callback = ema_callback
        
        self.cls_weight = cls_weight

        self.use_formula = use_formula
        self.use_fg = use_fg
        assert self.dynamics.use_fg == self.use_fg, f"Please check the setting of use_fg. (dynamics.use_fg: {self.dynamics.use_fg})" 
        if self.use_fg:
            # assert spec_fg_cls is not None, "Please check the input of spec_fg_cls which should be a loaded model."
            assert getattr(self.vae, "use_spec_cls_model", True), "Please check the input of vae model which should have a full spec_cls_model."
             
            if fix_spec_fg_cls:
                self.cls_weight = 0
                for param in self.vae.spec_cls_model.parameters():
                    param.requires_grad = False
                
            print(f"diff | fix_spec_fg_cls: {fix_spec_fg_cls} | cls_weight: {self.cls_weight}")
        

        self.save_hyperparameters(ignore=['vae', 'dynamics']) 
            


    def check_issues_norm_values(self, num_stdevs=8):
        zeros = torch.zeros((1, 1))
        gamma_0 = self.gamma(zeros)
        sigma_0 = self.sigma(gamma_0, target_tensor=zeros).item()

        # Checked if 1 / norm_value is still larger than 10 * standard
        # deviation.
        max_norm_value = max(self.norm_values[1], self.norm_values[2])

        if sigma_0 * num_stdevs > 1. / max_norm_value:
            raise ValueError(
                f'Value for normalization value {max_norm_value} probably too '
                f'large with sigma_0 {sigma_0:.5f} and '
                f'1 / norm_value = {1. / max_norm_value}')
    
    def instantiate_first_stage(self, vae: EnHierarchicaPosVAE):
        if not self.trainable_ae:
            self.vae = vae.eval()
            self.vae.train = disabled_train
            for param in self.vae.parameters():
                param.requires_grad = False
        else:
            self.vae = vae.train()
            for param in self.vae.parameters():
                param.requires_grad = True

    def sample_position_noise(self, n_samples, n_nodes, node_mask):
        """
        Samples mean-centered normal noise for z_x
        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
            size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        return z_x

    
    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)
    
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    
    def phi(self, spec_latent_vec, x, t, node_mask, edge_mask, context, fg_embed=None, spec_att_mask=None):

        if self.use_fg: assert fg_embed is not None
        if self.use_formula: assert spec_att_mask is not None

        net_out = self.dynamics._forward(spec_latent_vec, t, x, node_mask, edge_mask, context, fg_embed, spec_att_mask)
        # Predicted noise on positions
        net_out = net_out[:, :, :self.n_dims]
        
        return net_out

    
    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    
    def compute_error(self, net_out, gamma_t, eps):
        """Computes error, i.e. the most likely prediction of x."""
        eps_t = net_out
        if self.training and self.loss_type == 'l2':
            # denom = (self.n_dims + self.in_node_nf) * eps_t.shape[1]
            denom = self.n_dims * eps_t.shape[1]
            error = sum_except_batch((eps - eps_t) ** 2) / denom
        else:
            error = sum_except_batch((eps - eps_t) ** 2)
        return error
    
    def log_constants_p_x_given_z0(self, x, node_mask):
        """Computes p(x|z0)."""
        batch_size = x.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_x = (n_nodes - 1) * self.n_dims

        zeros = torch.zeros((x.size(0), 1), device=x.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_x * (- log_sigma_x - 0.5 * np.log(2 * np.pi))
    
    def subspace_dimensionality(self, node_mask):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        number_of_nodes = torch.sum(node_mask.squeeze(2), dim=1)
        return (number_of_nodes - 1) * self.n_dims
    
    def kl_prior(self, xh, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((xh.size(0), 1), device=xh.device)
        gamma_T = self.gamma(ones)
        alpha_T = self.alpha(gamma_T, xh)

        # Compute means.
        mu_T = alpha_T * xh
        mu_T_x, mu_T_h = mu_T[:, :, :self.n_dims], mu_T[:, :, self.n_dims:]

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.sigma(gamma_T, mu_T_x).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        sigma_T_h = self.sigma(gamma_T, mu_T_h)

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T_x), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(node_mask)
        kl_distance_x = gaussian_KL_for_dimension(mu_T_x, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x
        
        # Compute KL for h-part.
        zeros, ones = torch.zeros_like(mu_T_h), torch.ones_like(sigma_T_h)
        kl_distance_h = gaussian_KL(mu_T_h, sigma_T_h, zeros, ones, node_mask)
        return kl_distance_x + kl_distance_h
    
    def log_pxh_given_z0_without_constants(
            self, x, h, z_t, gamma_0, eps, net_out, node_mask, epsilon=1e-10):
        # Discrete properties are predicted directly from z_t.
        # z_h_cat = z_t[:, :, self.n_dims:-1] if self.include_charges else z_t[:, :, self.n_dims:]
        # z_h_int = z_t[:, :, -1:] if self.include_charges else torch.zeros(0).to(z_t.device)
        d_int = h['integer'].shape[-1]
        z_h_cat = z_t[:, :, self.n_dims:-d_int] if self.include_charges else z_t[:, :, self.n_dims:]
        z_h_int = z_t[:, :, -d_int:] if self.include_charges else torch.zeros(0).to(z_t.device)

        # Take only part over x.
        # eps_x = eps[:, :, :self.n_dims]
        # net_x = net_out[:, :, :self.n_dims]
        eps_x = eps
        net_x = net_out

        # Compute sigma_0 and rescale to the integer scale of the data.
        sigma_0 = self.sigma(gamma_0, target_tensor=z_t)
        sigma_0_cat = sigma_0 * self.norm_values[1]
        sigma_0_int = sigma_0 * self.norm_values[2]

        # Computes the error for the distribution N(x | 1 / alpha_0 z_0 + sigma_0/alpha_0 eps_0, sigma_0 / alpha_0),
        # the weighting in the epsilon parametrization is exactly '1'.
        log_p_x_given_z_without_constants = -0.5 * self.compute_error(net_x, gamma_0, eps_x)
        
        return log_p_x_given_z_without_constants


    
    def compute_loss(self, spec_embed, x, h, node_mask, edge_mask, context, t0_always, 
                     fg_embed=None, spec_att_mask=None, lowest_t_=None):
        """Computes an estimator for the variational lower bound, or the simple loss (MSE)."""

        # This part is about whether to include loss term 0 always.
        if t0_always:
            # loss_term_0 will be computed separately.
            # estimator = loss_0 + loss_t,  where t ~ U({1, ..., T})
            lowest_t = 1
        else:
            # estimator = loss_t,           where t ~ U({0, ..., T})
            lowest_t = 0
        
        if lowest_t_ is not None:
            lowest_t = lowest_t_
            
        # Sample a timestep t.
        t_int = torch.randint(
            lowest_t, self.T + 1, size=(x.size(0), 1), device=x.device).float()
        # print("t_int", t_int)
        s_int = t_int - 1
        t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.T
        t = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = self.inflate_batch_array(self.gamma(s), x)
        gamma_t = self.inflate_batch_array(self.gamma(t), x)

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.alpha(gamma_t, x)
        sigma_t = self.sigma(gamma_t, x)

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        # eps = self.sample_combined_position_feature_noise(
            # n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
        # Concatenate x, h[integer] and h[categorical].
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        # z_t = alpha_t * xh + sigma_t * eps

        # Sample z_{xt} ~ Normal(alpha_t x, sigma_t)
        eps = sample_center_gravity_zero_gaussian_with_mask(
            size=(x.size(0), x.size(1), self.n_dims), device=node_mask.device,
            node_mask=node_mask)
        z_xt = alpha_t * x + sigma_t * eps

        # assert_mean_zero_with_mask(z_t[:, :, :self.n_dims], node_mask)
        assert_mean_zero_with_mask(z_xt, node_mask)

        # Concatenate z_xt with h
        z_t = torch.cat([z_xt, h['categorical'], h['integer']], dim=2)

        # Neural net prediction.
        # net_out = self.phi(z_t, t, node_mask, edge_mask, context)
        net_out = self.phi(spec_embed, z_t, t, node_mask, edge_mask, context, fg_embed, spec_att_mask)

        # Compute the error.
        error = self.compute_error(net_out, gamma_t, eps)

        if self.training and self.loss_type == 'l2':
            SNR_weight = torch.ones_like(error)
        else:
            # Compute weighting with SNR: (SNR(s-t) - 1) for epsilon parametrization.
            SNR_weight = (self.SNR(gamma_s - gamma_t) - 1).squeeze(1).squeeze(1)
        assert error.size() == SNR_weight.size()
        loss_t_larger_than_zero = 0.5 * SNR_weight * error

        # The _constants_ depending on sigma_0 from the
        # cross entropy term E_q(z0 | x) [log p(x | z0)].
        neg_log_constants = -self.log_constants_p_x_given_z0(x, node_mask)

        # Reset constants during training with l2 loss.
        if self.training and self.loss_type == 'l2':
            neg_log_constants = torch.zeros_like(neg_log_constants)

        # The KL between q(z1 | x) and p(z1) = Normal(0, 1). Should be close to zero.
        kl_prior = self.kl_prior(xh, node_mask)

        # Combining the terms
        if t0_always:
            loss_t = loss_t_larger_than_zero
            num_terms = self.T  # Since t=0 is not included here.
            estimator_loss_terms = num_terms * loss_t

            # Compute noise values for t = 0.
            t_zeros = torch.zeros_like(s)
            gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
            alpha_0 = self.alpha(gamma_0, x)
            sigma_0 = self.sigma(gamma_0, x)

            # Sample z_0 given x, h for timestep t, from q(z_t | x, h)
            # eps_0 = self.sample_combined_position_feature_noise(
            #     n_samples=x.size(0), n_nodes=x.size(1), node_mask=node_mask)
            # z_0 = alpha_0 * xh + sigma_0 * eps_0
            eps_0 = sample_center_gravity_zero_gaussian_with_mask(
                size=(x.size(0), x.size(1), self.n_dims), device=node_mask.device,
                node_mask=node_mask)
            z_x0 = alpha_0 * x + sigma_0 * eps_0

            z_0 = torch.cat([z_x0, h['categorical'], h['integer']], dim=2)

            net_out = self.phi(spec_embed, z_0, t_zeros, node_mask, edge_mask, context, fg_embed, spec_att_mask)

            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_0, gamma_0, eps_0, net_out, node_mask)

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()
            assert kl_prior.size() == loss_term_0.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants + loss_term_0

        else:
            # Computes the L_0 term (even if gamma_t is not actually gamma_0)
            # and this will later be selected via masking.
            loss_term_0 = -self.log_pxh_given_z0_without_constants(
                x, h, z_t, gamma_t, eps, net_out, node_mask)

            t_is_not_zero = 1 - t_is_zero

            loss_t = loss_term_0 * t_is_zero.squeeze() + t_is_not_zero.squeeze() * loss_t_larger_than_zero

            # Only upweigh estimator if using the vlb objective.
            if self.training and self.loss_type == 'l2':
                estimator_loss_terms = loss_t
            else:
                num_terms = self.T + 1  # Includes t = 0.
                estimator_loss_terms = num_terms * loss_t

            assert kl_prior.size() == estimator_loss_terms.size()
            assert kl_prior.size() == neg_log_constants.size()

            loss = kl_prior + estimator_loss_terms + neg_log_constants

        assert len(loss.shape) == 1, f'{loss.shape} has more than only batch dim.'

        return loss, {'t': t_int.squeeze(), 'loss_t': loss.squeeze(),
                      'error': error.squeeze()}
    
    def log_constants_p_h_given_z0(self, h, node_mask):
        """Computes p(h|z0)."""
        batch_size = h.size(0)

        n_nodes = node_mask.squeeze(2).sum(1)  # N has shape [B]
        assert n_nodes.size() == (batch_size,)
        degrees_of_freedom_h = n_nodes * self.n_dims

        zeros = torch.zeros((h.size(0), 1), device=h.device)
        gamma_0 = self.gamma(zeros)

        # Recall that sigma_x = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
        log_sigma_x = 0.5 * gamma_0.view(batch_size)

        return degrees_of_freedom_h * (- log_sigma_x - 0.5 * np.log(2 * np.pi))
    
    def forward(self, spec, x, h, node_mask=None, edge_mask=None, context=None, formula_ids=None, lowest_t_=None, fg_onehot=None):
        """
        Computes the loss (type l2 or NLL) if training. And if eval then always computes NLL.
        args:
            lowest_t_: for controlling the times step when analyzing
        """
        # if not self.training: print("validation forward formula_ids", (formula_ids is None))
        if self.use_formula: 
            assert formula_ids is not None, "Please check the input of formula_ids if use_formula."
            z_h, spec_embed, spec_att_mask = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=formula_ids)
        else: 
            z_h, spec_embed, spec_att_mask = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=None)
        # if not self.training: print("validation forward spec_att_mask", (spec_att_mask is None))
        fg_embed = None
        if self.use_fg:
            assert formula_ids is not None, "Please check the input of formula_ids if use_fg"
            fg_embed = self.vae.spec_cls_model.decode(self.vae.spec_cls_model.fg_queries, spec_embed, spec_att_mask)
  
        # Encode data to latent space.
        z_x_mu, z_x_sigma = self.vae.encode(x, z_h, node_mask, edge_mask, context)
        # Compute fixed sigma values.
        t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
        gamma_0 = self.inflate_batch_array(self.gamma(t_zeros), x)
        sigma_0 = self.sigma(gamma_0, x)

        # Infer latent z.
        assert_correctly_masked(z_x_mu, node_mask)
        z_x_sigma = sigma_0
        z_x = self.vae.sample_normal(z_x_mu, z_x_sigma, node_mask)
        z_x = z_x.detach()
        assert_correctly_masked(z_x, node_mask)

        # Compute reconstruction loss.
        if self.trainable_ae:
            # Decoder output (reconstruction).
            zx_zh = torch.cat([z_x, z_h], dim=2)
            x_recon, _ = self.vae.decoder(zx_zh, node_mask, edge_mask, context)
            loss_recon = self.vae.compute_reconstruction_error(x_recon, x)
            if self.cls_weight > 0:
                cls_logits = self.vae.spec_cls_model.cls_head(fg_embed).squeeze(-1)
                loss_cls = self.vae.spec_cls_model._cal_loss(cls_logits, fg_onehot)
        else:
            loss_recon = 0

        assert_mean_zero_with_mask(z_x, node_mask)
        # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
        z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

        if self.training:
            # Only 1 forward pass when t0_always is False.
            loss_ld, loss_dict = self.compute_loss(
                spec_embed, z_x, z_h, node_mask, edge_mask, context, t0_always=False, 
                fg_embed=fg_embed, spec_att_mask=spec_att_mask, lowest_t_=lowest_t_)
        else:
            # Less variance in the estimator, costs two forward passes.
            # print("validation forward before compute_loss", (spec_att_mask is None))
            loss_ld, loss_dict = self.compute_loss(
                spec_embed, z_x, z_h, node_mask, edge_mask, context, t0_always=True, 
                fg_embed=fg_embed, spec_att_mask=spec_att_mask, lowest_t_=lowest_t_)
        
        neg_log_pxh = loss_ld + loss_recon
        if self.cls_weight == 0:
            return neg_log_pxh, {"loss_ld": loss_ld.detach().mean(), "loss_recon": loss_recon.detach().mean()}
        else:
            neg_log_pxh += loss_cls * self.cls_weight
            return neg_log_pxh, {"loss_ld": loss_ld.detach().mean(), "loss_recon": loss_recon.detach().mean(), "loss_cls": loss_cls.detach().mean()}

    
    
    def training_step(self, batch, batch_idx):
        #  training_step defines the train loop.
        # it is independent of forward
        x = batch['positions'].to(self.device_, self.dtype_)
        bs = x.shape[0]
        node_mask = batch['atom_mask'].to(self.device_, self.dtype_).unsqueeze(2)
        edge_mask = batch['edge_mask'].to(self.device_, self.dtype_)
        one_hot = batch['one_hot'].to(self.device_, self.dtype_)
        charges = (batch['charges'] if self.include_charges else torch.zeros(0)).to(self.device_, self.dtype_)
        if self.cls_weight > 0:
            fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        else:
            fg_onehot = None

        spec = batch["spectra"].to(self.device_, self.dtype_)
        if self.use_formula or self.use_fg:
            formula_ids = batch["formula"].int().to(self.device_)
        else:
            formula_ids = None

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
        loss, loss_dict = self(spec, x, h, node_mask, edge_mask, context=None, formula_ids=formula_ids, fg_onehot=fg_onehot)
        loss = loss.mean()
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, batch_size=bs)
        self.log("train_loss_ld", loss_dict["loss_ld"], batch_size=bs)
        self.log("train_loss_recon", loss_dict["loss_recon"], batch_size=bs)
        if self.cls_weight > 0:
            self.log("train_loss_cls", loss_dict["loss_cls"], batch_size=bs)
        

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

        if self.cls_weight > 0:
            fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        else:
            fg_onehot = None
        
        spec = batch["spectra"].to(self.device_, self.dtype_)
        if self.use_formula or self.use_fg:
            formula_ids = batch["formula"].int().to(self.device_)
        else:
            formula_ids = None
        # print("validation step get formula: ", (formula_ids is None))
        # if formula_ids is None: print( batch["formula"])

        x = remove_mean_with_mask(x, node_mask)
        
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
        self.log("val_loss_ld", loss_dict["loss_ld"], batch_size=bs)
        self.log("val_loss_recon", loss_dict["loss_recon"], batch_size=bs)
        if self.cls_weight > 0:
            self.log("val_loss_cls", loss_dict["loss_cls"], batch_size=bs)

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
        spec = batch["spectra"].to(self.device_, self.dtype_)

        if self.cls_weight > 0:
            fg_onehot = batch["func_groups"].to(self.device_, torch.float)
        else:
            fg_onehot = None

        if self.use_formula or self.use_fg:
            formula_ids = batch["formula"].int().to(self.device_)
        else:
            formula_ids = None
        
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
        self.log("test_loss_ld", loss_dict["loss_ld"], batch_size=bs)
        self.log("test_loss_recon", loss_dict["loss_recon"], batch_size=bs)
        if self.cls_weight > 0:
            self.log("test_loss_cls", loss_dict["loss_cls"], batch_size=bs)
        

        return loss

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=1e-12
        )

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
    
    def unnormalize(self, x, h_cat, h_int, node_mask):
        x = x * self.norm_values[0]
        h_cat = h_cat * self.norm_values[1] + self.norm_biases[1]
        h_cat = h_cat * node_mask
        h_int = h_int * self.norm_values[2] + self.norm_biases[2]

        if self.include_charges:
            h_int = h_int * node_mask

        return x, h_cat, h_int
    
    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        # eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
        eps = self.sample_position_noise(bs, mu.size(1), node_mask)
        return mu + sigma * eps
    
    
    
    def sample_p_zs_given_zt(self, spec, z_h, s, t, z_xt, node_mask, edge_mask, context, fix_noise=False,
                             fg_embed=None, spec_att_mask=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_xt)

        sigma_s = self.sigma(gamma_s, target_tensor=z_xt)
        sigma_t = self.sigma(gamma_t, target_tensor=z_xt)

        # Neural net prediction.
        zt = torch.cat([z_xt, z_h], dim=2)
        eps_t = self.phi(spec, zt, t, node_mask, edge_mask, context,
                         fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        # Compute mu for p(zs | zt).
       
        assert_mean_zero_with_mask(z_xt, node_mask)
        assert_mean_zero_with_mask(eps_t, node_mask)

        mu = z_xt / alpha_t_given_s - (sigma2_t_given_s / alpha_t_given_s / sigma_t) * eps_t

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the paramters derived from zt.
        zs = self.sample_normal(mu, sigma, node_mask, fix_noise)

       
        assert zs.shape[2] == self.n_dims, f"Check zs shape ({zs.shape})."
        zs = remove_mean_with_mask(zs, node_mask)
        return zs
    
    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        if self.parametrization == 'x':
            x_pred = net_out
        elif self.parametrization == 'eps':
            sigma_t = self.sigma(gamma_t, target_tensor=net_out)
            alpha_t = self.alpha(gamma_t, target_tensor=net_out)
            eps_t = net_out
            x_pred = 1. / alpha_t * (zt - sigma_t * eps_t)
        else:
            raise ValueError(self.parametrization)

        return x_pred
    
    def sample_p_x_given_zx0(self, spec, z_h, z_x0, node_mask, edge_mask, context, fix_noise=False, sample_mode="sample",
                             fg_embed=None, spec_att_mask=None):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z_x0.size(0), 1), device=z_x0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)

        z0 = torch.cat([z_x0, z_h], dim=2)
        net_out = self.phi(spec, z0, zeros, node_mask, edge_mask, context,
                           fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        # Compute mu for p(zs | zt).
        mu_x = self.compute_x_pred(net_out, z_x0, gamma_0)
        # xh = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        if sample_mode == "sample":
            x = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)
        elif sample_mode == "only_mean":
            x = mu_x

        x = x * self.norm_values[0]
        return x

   
    
    @torch.no_grad()
    def _sample(self, spec_embed, z_h, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, sample_mode="sample",
                fg_embed=None, spec_att_mask=None):
        """
        Draw samples from the generative model.
        """
        if fix_noise:
            # Noise is broadcasted over the batch axis, useful for visualizations.
            # z = self.sample_combined_position_feature_noise(1, n_nodes, node_mask)
            z_x = sample_center_gravity_zero_gaussian_with_mask(
                size=(1, n_nodes, self.n_dims), device=node_mask.device,
                node_mask=node_mask)
        else:
            # z = self.sample_combined_position_feature_noise(n_samples, n_nodes, node_mask)
            z_x = sample_center_gravity_zero_gaussian_with_mask(
                size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
                node_mask=node_mask)

        # assert_mean_zero_with_mask(z[:, :, :self.n_dims], node_mask)
        assert_mean_zero_with_mask(z_x, node_mask)


        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_x.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z_x = self.sample_p_zs_given_zt(spec_embed, z_h, s_array, t_array, z_x, node_mask, edge_mask, context, fix_noise=fix_noise,
                                            fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        # Finally sample p(x, h | z_0).
        x = self.sample_p_x_given_zx0(spec_embed, z_h, z_x, node_mask, edge_mask, context, fix_noise=fix_noise, sample_mode=sample_mode,
                                      fg_embed=fg_embed, spec_att_mask=spec_att_mask)
   
        assert_mean_zero_with_mask(x, node_mask)

        max_cog = torch.sum(x, dim=1, keepdim=True).abs().max().item()
        if max_cog > 5e-2:
            print(f'Warning cog drift with error {max_cog:.3f}. Projecting '
                  f'the positions down.')
            x = remove_mean_with_mask(x, node_mask)

        return x
    
    @torch.no_grad()
    def sample(self, spec, h, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise=False, sample_mode="sample",
               formula_ids=None):
        """
        Draw samples from the generative model.
        """
        # z_h, spec_embed = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=formula_ids)
        if self.use_formula: 
            assert formula_ids is not None, "Please check the input of formula_ids if use_formula."
            z_h, spec_embed, spec_att_mask = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=formula_ids)
        else: 
            z_h, spec_embed, spec_att_mask = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=None)
        # if not self.training: print("validation forward spec_att_mask", (spec_att_mask is None))
        fg_embed = None
        if self.use_fg:
            assert formula_ids is not None, "Please check the input of formula_ids if use_fg"
            _, fg_embed = self.vae.spec_cls_model(spec, formula_ids)

        # Denoise to obtain z_x
        z_x = self._sample(spec_embed, z_h, n_samples, n_nodes, node_mask, edge_mask, context, fix_noise, sample_mode,
                           fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        # z_xh = torch.cat([z_x, z_h], dim=2)
        assert_correctly_masked(z_x, node_mask)
        x = self.vae.decode(z_x, z_h, node_mask, edge_mask, context)

        return x
   
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s
    
    @torch.no_grad()
    def _sample_chain(self, spec_embed, z_h, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None,
                      fg_embed=None, spec_att_mask=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        Args:
            keep_frames: int, the number of frames that will be kept. If `None`, keep all frames.

        """
        z_x = sample_center_gravity_zero_gaussian_with_mask(
                size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
                node_mask=node_mask)

        assert_mean_zero_with_mask(z_x[:, :, :self.n_dims], node_mask)

        if keep_frames is None:
            keep_frames = self.T
        else:
            assert keep_frames <= self.T
        chain = torch.zeros((keep_frames,) + z_x.size(), device=z_x.device)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s in reversed(range(0, self.T)):
            s_array = torch.full((n_samples, 1), fill_value=s, device=z_x.device)
            t_array = s_array + 1
            s_array = s_array / self.T
            t_array = t_array / self.T

            z_x = self.sample_p_zs_given_zt(
                spec_embed, z_h, s_array, t_array, z_x, node_mask, edge_mask, context,
                fg_embed=fg_embed, spec_att_mask=spec_att_mask)

            assert_mean_zero_with_mask(z_x[:, :, :self.n_dims], node_mask)

            # Write to chain tensor.
            write_index = (s * keep_frames) // self.T

            chain[write_index] = z_x
            # unnormalize
            # chain[write_index] = self.unnormalize_z(z_x, node_mask)
            # chain[write_index] = z_x * self.norm_values[0]
            
            

        # Finally sample p(x, h | z_0).
        x = self.sample_p_x_given_zx0(spec_embed, z_h, z_x, node_mask, edge_mask, context,
                                      fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        assert_mean_zero_with_mask(x[:, :, :self.n_dims], node_mask)

        x_ = x.unsqueeze(0)
        chain = torch.concat([x_, chain], dim=0)

        return chain  # (frames, batch_size, max_n_nodes, n_dims)
        
    
    @torch.no_grad()
    def sample_chain(self, spec, h, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=None,
                     formula_ids=None):
        """
        Draw samples from the generative model, keep the intermediate states for visualization purposes.
        """
        print("sample_chain_ff")
        print('use_formula', self.use_formula, 'formula is not None', (formula_ids is not None))
        z_h, spec_embed, spec_att_mask = self.vae.get_z_h_spec_embed(spec, h, node_mask, formula_ids=formula_ids)
        fg_embed = None
        if self.use_fg:
            assert formula_ids is not None, "Please check the input of formula_ids if use_fg"
            _, fg_embed = self.vae.spec_cls_model(spec, formula_ids)
        print('use_fg', self.use_fg, 'fg_embed is not None', (fg_embed is not None))

        chain = self._sample_chain(spec_embed, z_h, n_samples, n_nodes, node_mask, edge_mask, context, keep_frames,
                                   fg_embed=fg_embed, spec_att_mask=spec_att_mask)

        keep_frames = chain.size(0)
        chain_decoded = torch.zeros_like(chain)
        
        for j in range(keep_frames):
            z_x_j = chain[j]
            assert_mean_zero_with_mask(z_x_j[:, :, :self.n_dims], node_mask)

            x = self.vae.decode(z_x_j, z_h, node_mask, edge_mask, context)
            chain_decoded[j] = x
        
        return chain_decoded

    