import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import lightning as L
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Optional
import math

class BaseVAE(nn.Module, ABC):
    def __init__(
        self,
        input_dim: int,
        enc_hidden_dims: List[int],
        latent_dim: int,
        dec_hidden_dims: List[int],
        output_split_sections: List[int],
        split_section_types: List[str],
        loss_map: Dict,
        beta_kl: float = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.enc_hidden_dims = enc_hidden_dims
        self.latent_dim = latent_dim
        self.dec_hidden_dims = dec_hidden_dims
        self.parse_loss_map(loss_map)
        for split_section_type in split_section_types:
            assert split_section_type in loss_map.keys(), f'all values of split_section_types must be referenced in loss_map ({split_section_type} is missing)'
            assert split_section_type in ('score', 'distribution', 'categorical', 'numerical', 'multiclass'), f'all values of split_section_types must be one of (score, distribution, categorical, numerical))'
        assert len(output_split_sections) == len(split_section_types), f'length of output_split_sections({len(output_split_sections)}) must be same as length of split_section_types ({len(split_section_types)})'
        assert len(loss_map) == 1 if len(output_split_sections) == 0 else True, 'if output_split_sections is empty, exactly one loss must be provided in loss_map '
        assert len(split_section_types) > 0, 'split_section_types must always be provided (len >= 1)'
        self.output_split_sections = output_split_sections
        self.split_section_types = split_section_types
        self.beta_kl = beta_kl

        # Build encoder and decoder
        self.encoder_layers = self._build_encoder()
        self.decoder_layers = self._build_decoder()
        
        # Initialize distribution-specific layers in child classes
        self._build_distribution_layers()

    def parse_loss_map(self, loss_map):
        for key in loss_map:
            assert loss_map[key] in ('mse', 'cross_entropy', 'symmetric_kl', 'binary_cross_entropy') or callable(loss_map[key], ), 'All values in loss_map must be one of (mse, cross_entropy, symmetric_kl, binary_cross_entropy)'
            if loss_map[key] == 'mse':
                loss_map[key] = self.mse_loss
            elif loss_map[key] == 'cross_entropy':
                loss_map[key] = self.cross_entropy_loss
            elif loss_map[key] == 'symmetric_kl':
                loss_map[key] = self.symmetric_kl_loss
            elif loss_map[key] == 'binary_cross_entropy':
                loss_map[key] = self.binary_cross_entropy_loss
        self.loss_map = loss_map

    def set_beta_kl(self, beta_kl: float):
        """Allow dynamic KL weighting factor updates during training"""
        self.beta_kl = beta_kl

    def _build_encoder(self) -> nn.Module:
        """Builds the shared encoder architecture"""
        layers = []
        in_dim = self.input_dim
        
        # Build hidden layers
        for hidden_dim in self.enc_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
            
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Module:
        """Builds the shared decoder architecture"""
        layers = []
        hidden_dims = self.dec_hidden_dims
        
        # First layer from latent space
        in_dim = self.get_decoded_latent_dim()
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
            
        # Final output layer
        layers.extend([
            nn.Linear(in_dim, self.input_dim)
        ])
            
        return nn.Sequential(*layers)

    @abstractmethod
    def _build_distribution_layers(self):
        """Build the distribution-specific layers (implemented by child classes)"""
        pass

    @abstractmethod
    def get_decoded_latent_dim(self) -> int:
        """Return the dimensionality of the decoded latent vector"""
        pass

    @abstractmethod
    def encode_distribution(self, hidden: torch.Tensor) -> Any:
        """Convert encoder hidden state to distribution parameters"""
        pass

    @abstractmethod
    def reparameterize(self, *args) -> torch.Tensor:
        """Sample from the distribution using reparameterization"""
        pass

    @abstractmethod
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, *args) -> Dict[str, torch.Tensor]:
        """Calculate VAE loss components"""
        pass

    def mse_loss(self, x_recon, x, *args):
        loss = F.mse_loss(x_recon, x, reduction='none')
        return loss.sum(-1).mean()

    def binary_cross_entropy_loss(self, x_recon, x, *args):
        loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='mean')  # mean over batch (feature dim is already handled by cross_entropy)
        return loss

    def cross_entropy_loss(self, x_recon, x, *args):
        loss = F.cross_entropy(x_recon, x, reduction='mean')  # mean over batch (feature dim is already handled by cross_entropy)
        return loss

    def symmetric_kl_loss(self, x_recon, x, split_type):
        if split_type == 'score':
            # consider each value separately. Should still work with same kl_div as kl_div is calculated pointwise, and we sum it up anyway over all feature dimensions (bachwise)
            x_recon = F.logsigmoid(x_recon)
        else:
            x_recon = F.log_softmax(x_recon, dim=-1)
        x = torch.clamp(x, 0.01, 0.99)
        x = x.log()
        loss = 0.5*(F.kl_div(x_recon, x, reduction='batchmean', log_target = True) + F.kl_div(x, x_recon, reduction='batchmean', log_target = True))
        return loss

    def recon_loss_sections(self, x_recon, x, return_indvl_losses = False):
        """
        Handles different types of input variables so that different losses can be applied section-wise.
        """
        if len(self.output_split_sections) > 0:
            split_x = list(torch.split(x, self.output_split_sections, dim=1))
            split_x_recon = list(torch.split(x_recon, self.output_split_sections, dim=1))
            section_losses = [
                self.loss_map[split_type](x_recon_section, x_section, split_type)
                for x_recon_section, x_section, split_type in zip(split_x_recon, split_x, self.split_section_types)
            ]
            if return_indvl_losses:
                return section_losses
            else:
                recon_loss = torch.sum(torch.stack(section_losses))
        else:
            recon_loss = list(self.loss_map.values())[0](x_recon, x)
        return recon_loss

    def encode(self, x: torch.Tensor) -> Any:
        """Full encoding process without reparametrization"""
        hidden = self.encoder_layers(x)
        return self.encode_distribution(hidden)

    @abstractmethod
    def extract_embeddings(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Return embeddings"""
        pass

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to reconstructions"""
        return self.decoder_layers(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Forward pass through the VAE"""
        distribution_params = self.encode(x)
        z = self.reparameterize(*distribution_params)
        x_recon = self.decode(z)
        return x_recon, distribution_params


class GaussianVAE(BaseVAE):
    def _build_distribution_layers(self):
        """Build layers for Gaussian distribution parameters"""
        last_hidden = self.enc_hidden_dims[-1] if len(self.enc_hidden_dims) > 0 else self.input_dim
        self.fc_mu = nn.Linear(last_hidden, self.latent_dim)
        self.fc_logvar = nn.Linear(last_hidden, self.latent_dim)

    def get_decoded_latent_dim(self) -> int:
        """Gaussian latent space has same dim as latent_dim"""
        return self.latent_dim

    def encode_distribution(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Gaussian distribution parameters"""
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Gaussian reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        recon_loss = self.recon_loss_sections(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        total_loss = recon_loss + self.beta_kl*kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings"""
        return self.encode(x)[0]  #means

class CategoricalVAE(BaseVAE):
    def __init__(
        self,
        input_dim: int,
        enc_hidden_dims: List[int],
        latent_dim: int,
        dec_hidden_dims: List[int],
        output_split_sections: List[int],
        split_section_types: List[str],
        loss_map: Dict,
        beta_kl: float = 1,
        num_categories: int = 10,
        temperature: float = 1,
    ):
        self.num_categories = num_categories
        self.temperature = temperature
        super().__init__(input_dim, enc_hidden_dims, latent_dim, dec_hidden_dims, output_split_sections, split_section_types, loss_map, beta_kl)

    def set_temperature(self, temperature: float):
        """Allow dynamic temperature updates during training"""
        self.temperature = temperature

    def _build_distribution_layers(self):
        """Build layers for Categorical distribution parameters"""
        last_hidden = self.enc_hidden_dims[-1] if len(self.enc_hidden_dims) > 0 else self.input_dim
        self.fc_logits = nn.Linear(last_hidden, self.latent_dim * self.num_categories)

    def get_decoded_latent_dim(self) -> int:
        """Categorical latent space has dim = latent_dim * num_categories after one-hot"""
        return self.latent_dim * self.num_categories

    def encode_distribution(self, hidden: torch.Tensor) -> Tuple[torch.Tensor,]:
        """Return Categorical distribution parameters"""
        logits = self.fc_logits(hidden)
        return logits.view(-1, self.latent_dim, self.num_categories),

    def reparameterize(self, logits: torch.Tensor, return_indices=False) -> torch.Tensor:
        """
        Sample from categorical distribution using the Gumbel-Softmax trick
    
        During training: Use Gumbel-Softmax with temperature for differentiable sampling
        During inference: Use straight-through one-hot encoding
        """
        if self.training:
            gumbel = -torch.empty_like(logits).exponential_().log()  #exponential_: draws from exponential distribution, equivalent to -log(u), where u is uniform distributed
            gumbel = (logits + gumbel) / self.temperature
            y = F.softmax(gumbel, dim=-1)
            return y.reshape(y.size(0), -1)  # Flatten for decoder
        else:
            # Hard encoding during inference
            _, indices = torch.max(logits, dim=-1)
            if return_indices:
                return indices
            else:
                one_hot = F.one_hot(indices, num_classes=self.num_categories)
                return one_hot.view(logits.size(0), -1).float()

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        recon_loss = self.recon_loss_sections(x_recon, x)
        
        q = F.softmax(logits, dim=-1)
        log_q = F.log_softmax(logits, dim=-1)
        log_p = torch.log(torch.ones_like(q) / self.num_categories)
        kl_loss = torch.sum(q * (log_q - log_p))
        
        total_loss = recon_loss + self.beta_kl*kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings"""
        x = self.encode(x)[0]
        return x.flatten(1)  #flatten categorical logits
    

class PoissonVAE(BaseVAE):
    def __init__(
        self,
        input_dim: int,
        enc_hidden_dims: List[int],
        latent_dim: int,
        dec_hidden_dims: List[int],
        output_split_sections: List[int],
        split_section_types: List[str],
        loss_map: Dict,
        beta_kl: float = 1,
        temperature: float = 0.5,
        compute_n_exp_numerically = False
    ):
        super().__init__(input_dim, enc_hidden_dims, latent_dim, dec_hidden_dims, output_split_sections, split_section_types, loss_map, beta_kl)
        self.logr = nn.Parameter(torch.randn(latent_dim, requires_grad=True))  #todo: verify if initialization is correct or should be modified
        self.temperature = temperature  #todo: should be annealed, think where to do this
        self.compute_n_exp_numerically = compute_n_exp_numerically  # if false, uses heuristic for n_exp (faster)

    def set_temperature(self, temperature: float):
        """Allow dynamic temperature updates during training"""
        self.temperature = temperature

    def _build_distribution_layers(self):
        """Build layers for Poisson distribution parameters"""
        last_hidden = self.enc_hidden_dims[-1] if len(self.enc_hidden_dims) > 0 else self.input_dim
        self.fc_logdr = nn.Linear(last_hidden, self.latent_dim)

    def get_decoded_latent_dim(self) -> int:
        """Poisson latent space has same dim as latent_dim"""
        return self.latent_dim

    def encode_distribution(self, hidden: torch.Tensor) -> Tuple[torch.Tensor]:
        """Return Poisson distribution parameters"""
        logdr = self.fc_logdr(hidden)
        return logdr,

    def get_n_exp(self, lam_max):
        if self.compute_n_exp_numerically:
            # Compute adaptive number of samples using inverse CDF
            poisson_dist = dist.Poisson(max(lam_max, 1))
            #icdf not implemented for poisson in pytorch...
            upper_bound = math.ceil(lam_max + 6 * math.sqrt(lam_max))
            k = torch.arange(0, upper_bound + 1, dtype=torch.float)
            log_cdf = torch.cumsum(poisson_dist.log_prob(k).exp(), dim=0)
            target_prob = 0.99999
            quantile_index = torch.argmax((log_cdf >= target_prob).float())
            if quantile_index > 0:
                n_exp = int(k[quantile_index])
            else:
                n_exp = upper_bound + 1
            self.n_exp = n_exp
            self.lam_max = lam_max
            return n_exp
        else:
            return math.ceil(lam_max + 5 * math.sqrt(lam_max))  #approximate value for 0.99999

    def reparameterize(self, logdr, temperature=1.0):
        """
        Reparameterized sampling for Poisson distribution
        
        Args:
        - logdr (torch.Tensor): Log of rate delta parameter, shape [B, K]
        - n_exp (int): Number of exponential samples to generate
        - temperature (float): Controls the sharpness of thresholding
        
        Returns:
        - z (torch.Tensor): Event counts, shape [B, K]
        """
        lam = (logdr + self.logr).exp()
        if self.training:
            # Find maximum rate in each batch
            lam_max = lam.max().item()
            n_exp = self.get_n_exp(lam_max)

            # Create exponential distribution
            exp_dist = torch.distributions.Exponential(lam)
            
            # Sample inter-event times
            delta_t = exp_dist.rsample((n_exp,))  # Shape: [n_exp, B, K]
            
            # Compute arrival times
            times = torch.cumsum(delta_t, dim=0)  # Shape: [n_exp, B, K]
            
            # Soft indicator for events within unit time
            indicator = torch.sigmoid((1 - times) / temperature)  # Shape: [n_exp, B, K]
            
            # Compute event counts
            z = torch.sum(indicator, dim=0)  # Shape: [B, K]
        
        else:
            z = lam.round()
        
        return z

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        logdr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        recon_loss = self.recon_loss_sections(x_recon, x)
        dr = logdr.exp()
        kl_loss = torch.sum(self.logr.exp()*(1 - dr + dr*logdr), dim=1).mean()
        total_loss = recon_loss + self.beta_kl*kl_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings"""
        return self.encode(x)[0]  #logdr
    
"""
Only required for baseline of prediction quality using input instead of embeddings.
Does nothing (returns input as embedding)
"""
class IdentityVAE(BaseVAE):

    def get_decoded_latent_dim(self) -> int:
        return self.input_dim
    
    def _build_distribution_layers(self):
        pass

    def encode_distribution(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ]:
        return hidden,

    def reparameterize(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, *args) -> dict[str, torch.Tensor]:
        pass
    
"""
Lightning Wrapper
"""
class LitVAE(L.LightningModule):
    def __init__(
        self,
        model_name,
        model_config,
        learning_rate: float = 1e-3,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay_rate: float = 0.05,
        initial_beta_kl: float = 0.01,
        beta_kl_max: float = 1 ,
        beta_kl_growth_rate: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # Store temperature parameters
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay_rate = temperature_decay_rate
        # Current temperature (will be updated during training)
        self.current_temperature = initial_temperature
        if 'temperature' not in model_config and model_name in ('CategoricalVAE','PoissonVAE'):
            model_config['temperature'] = self.initial_temperature

        # Store beta_kl
        self.initial_beta_kl = initial_beta_kl
        self.beta_kl_max = beta_kl_max
        self.beta_kl_growth_rate = beta_kl_growth_rate
        self.current_beta_kl = self.initial_beta_kl
        if 'beta_kl' not in model_config:
            model_config['beta_kl'] = self.initial_beta_kl

        if model_name == 'GaussianVAE':
            self.vae = GaussianVAE(**model_config)
        elif model_name == 'CategoricalVAE':
            self.vae = CategoricalVAE(**model_config)
        elif model_name == 'PoissonVAE':
            self.vae = PoissonVAE(**model_config)
        elif model_name == 'IdentityVAE':
            self.vae = IdentityVAE(**model_config)
        else:
            assert False, 'model_name must be one of (GaussianVAE, CategoricalVAE, PoissonVAE)'

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch

        # Update VAE's temperature if it's a CategoricalVAE
        if isinstance(self.vae, CategoricalVAE) or isinstance(self.vae, PoissonVAE):
            self.vae.set_temperature(self.current_temperature)

        # Update beta_kl
        self.vae.set_beta_kl(self.current_beta_kl)

        x_recon, distribution_params = self.vae(client_features)
        losses = self.vae.loss_function(client_features, x_recon, *distribution_params)

        # Log temperature
        self.log('temperature', self.current_temperature, on_step=False, on_epoch=True, batch_size=client_features.shape[0])
        # Log beta_kl
        self.log('beta_kl', self.current_beta_kl, on_step=False, on_epoch=True, batch_size=client_features.shape[0])

        # Log all loss components
        for name, value in losses.items():
            if name=='total_loss':
                self.log(f'train_{name}', value, on_step=True, on_epoch=True, prog_bar=True, batch_size=client_features.shape[0])
            else:
                self.log(f'train_{name}', value, on_step=False, on_epoch=True, prog_bar=False, batch_size=client_features.shape[0])
        return losses['total_loss']
    
    def on_train_epoch_end(self):
        # Anneal temperature using exponential decay
        self.current_temperature = max(
            self.min_temperature, 
            self.initial_temperature * math.exp(-self.temperature_decay_rate * self.current_epoch)
        )
        # Anneal beta_kl using exponential growth
        self.current_beta_kl = min(
            self.beta_kl_max, 
            self.initial_beta_kl * math.exp(self.beta_kl_growth_rate * self.current_epoch)
        )
    
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch
        x_recon, distribution_params = self.vae(client_features)
        losses = self.vae.loss_function(client_features, x_recon, *distribution_params)
        # Log all loss components
        for name, value in losses.items():
            if name=='total_loss':
                self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=True, batch_size=client_features.shape[0])
            else:
                self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=False, batch_size=client_features.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)