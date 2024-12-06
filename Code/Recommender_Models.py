# -*- coding: utf-8 -*-
"""

"""
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.classification import AUROC, Accuracy, MultilabelRankingAveragePrecision


######################################################################################
## BaseAssetRecommender using a masking strategy
######################################################################################
class BaseAssetRecommender(nn.Module):
    """
    Base Class for Asset Recommendation for given portfolios
    """
    def __init__(
            self,
            num_assets,
            asset_key_index_map,
            probs_norm_method,
            asset_counts,
            r_mask = 1,
            masking_strategy = 'random'):
        super().__init__()
        self.num_assets = num_assets
        self.set_asset_key_index_map(asset_key_index_map)
        asset_priors = asset_counts/asset_counts.sum()
        self.register_buffer('asset_prior_evidence', (asset_priors/(1-asset_priors)).log())
        if probs_norm_method == 'softmax':
            self.scores_to_prob = nn.Softmax(dim=1)
        elif probs_norm_method == 'sigmoid':
            self.scores_to_prob = nn.Sigmoid()
        else:
            assert False, 'probs_norm_method must be one of (softmax, sigmoid)'
        if masking_strategy in ('random', 'all_combinations', 'random_sequential'):
            self.masking_strategy = masking_strategy
            self.r_mask = r_mask
        else:
            assert False, 'masking_strategy must be one of (random, all_combinations, random_sequential)'

    def set_asset_key_index_map(self, asset_key_index_map):
        """
        Used to store the mapping of asset keys to embedding indices. Needs to be called after loading the model
        """
        assert isinstance(asset_key_index_map, dict), f'asset_key_index_map must be a dictionary)'
        # check if values in asset_key_index_map are consecutive indices starting from 0
        sorted_k_v_list = sorted(asset_key_index_map.items(), key=lambda item: item[1])
        asset_key_index_map = {k: v for k,v in sorted_k_v_list}
        values = list(asset_key_index_map.values())
        value_checks = [values[i+1]-curr_val in (0,1) for i,curr_val in enumerate(values[:-1])]   # allow for duplicates (e.g. padding_index)
        assert values[0]==0 and all(value_checks), 'asset_key_index_map must contain values that correspond to indices starting with 0, i.e. every index between 0 and a given maximum must be included'
        # store dictionaries in model
        self.asset_key_index_map = asset_key_index_map
        self.target_asset_keys = list(asset_key_index_map.keys())

    def asset_key_to_index(self, asset_keys):
        indices = self.asset_key_index_map[asset_keys]
        return torch.tensor(indices)
    
    def forward(self, asset_indices, portfolio_indices, return_indvl_scores=False):
        # to be overridden in individual model classes. Returns scores (not normalized) for each asset
        # return unnomalized scores (i.e. no softmax or sigmoid)
        return None

    def random_masking(self, portfolio_indices, r=1, masking_strategy='random'):
        """
        Perform per-portfolio random masking. This assumes that portfolios are already pre-filtered
        to exclude portfolios with low number of assets, such that the masking of one or several assets
        cannot produce a portfolio with no remaining assets.
        Returns:
         - Mask: 1=Masked, 0=Not Masked
        masking_strategies:
            - random: Simply mask r assets per portfolio. asset_indices and portfolio_indices remain the same.
                If r is a float between 0 and 1, mask out this percentage of total assets per portfolio
            - all_combinations: Use all combinations of length r. 
              asset_indices and portfolio_indices are repeated accordingly (batch size increases)
            - random_sequential: Use random ordering, then predict each asset at a time based on the previous ones.
              asset_indices and portfolio_indices are repeated accordingly (batch size increases)
        """
        portf, portf_counts = torch.unique(portfolio_indices, sorted=False, return_counts=True)  #this works because portfolio_indices is already sorted
        portf_start_indices = torch.cumsum(portf_counts, dim=0)
        noise = torch.rand(portf.shape[0], portf_counts.max(), device=portfolio_indices.device)
        mask = torch.zeros_like(portfolio_indices)
        if masking_strategy == 'random':
            # mask out a single random asset per portfolio
            if isinstance(r,int):
                mask_indices = torch.argsort(noise)[:,:r]
            elif isinstance(r,float) and 0 < r < 1:
                assert False, 'Not implemented yet'
            else:
                assert False, 'value of r ({r}) not allowed'
            # limit to number of assets per portfolio
            mask_indices = torch.remainder(mask_indices, portf_counts.unsqueeze(-1))   #todo: this can give less than r values if r>1, use unique indices
            # add start index of portfolio to get indices for mask
            mask_indices[1:] += portf_start_indices[:-1].unsqueeze(-1)
            mask[mask_indices.flatten()] = 1
        elif masking_strategy == 'batch_random_range':
            #todo: r is list of min and max values. Sample r, then apply random masking over full batch,
            #then unmask again for portfolios with unmasked assets under min_unmasked_assets, and additionally
            #mask for portfolios with masked assets below min_masked_assets
            assert False, 'Not implemented yet'
        elif masking_strategy in ('all_combinations', 'random_sequential'):
            assert False, 'Not implemented yet'
        else:
            assert False, 'masking_strategy must be one of (random, batch_random_range, all_combinations, random_sequential)'

        return mask.bool()
    
    def forward_masking(self, asset_indices, portfolio_indices, ign_self_scores=True):
        """
        Makes a forward pass by masking out one or several assets in each portfolio, and then predicts the missing assets.
        ign_self_scores:    If set to true, set the scores of the assets that are included in the portfolio to -inf, so that they will have 0 loss. 
                            This is relevant, because for our current masking strategy, the targets are always strictly different than the non-masked
                            assets, so without this setting the model is incentivized to give very low scores to the non-masked assets in the portfolio,
                            which can have a negative influence for the embeddings/weigths and is not aligned with our objective and our later use case.
        """
        mask = self.random_masking(portfolio_indices=portfolio_indices, r=self.r_mask, masking_strategy=self.masking_strategy)
        target = torch.zeros(portfolio_indices.max()+1, self.num_assets).to(asset_indices.device)
        target[torch.masked_select(portfolio_indices, mask), torch.masked_select(asset_indices, mask)] = 1.
        mask = ~mask
        unmasked_asset_indices = torch.masked_select(asset_indices, mask)
        unmasked_portfolio_indices = torch.masked_select(portfolio_indices, mask)
        scores = self(unmasked_asset_indices, unmasked_portfolio_indices, return_indvl_scores=False)
        scores = scores + self.asset_prior_evidence.unsqueeze(0)
        if ign_self_scores:
            scores_log_adj = torch.zeros_like(scores)
            scores_log_adj[torch.masked_select(portfolio_indices, mask), torch.masked_select(asset_indices, mask)] = -1e30
            scores = scores + scores_log_adj
        return scores, target

    def get_portfolio_pred_scores(self, portfolio_asset_list, return_indvl_scores=False, return_probs=False, return_target_asset_keys=False, fwd_kwargs={}):
        """
        portfolio_asset_list: list of lists, where each element in first list represents a portfolio given
        by a list of asset keys
        Returns scores for each target asset
            - if return_probs, returns normalised scores in range (0,1)
            - if return_indvl_scores=False: returns list of scores for target assets per portfolio, so matrix of shape (num_portfolios, num_target_assets)
            - if return_target_asset_keys, returns the keys (ISIN) of the predicted assets as well (same order as predictions)
        """
        assert (isinstance(portfolio_asset_list, list) or isinstance(portfolio_asset_list, tuple)) and (isinstance(portfolio_asset_list[0], list) or isinstance(portfolio_asset_list, tuple)), 'portfolio_asset_list must be a list of lists, i.e. list of portfolios with list of asset keys (ISIN) per portfolio'
        assert hasattr(self, 'asset_key_index_map'), 'asset_key_index_map is not set. Model needs to be initialized by calling model.set_asset_key_index_map'
        asset_indices = []
        portf_index = 0
        portfolio_indices = []
        for portf in portfolio_asset_list:
            asset_indices = self.asset_key_to_index[portf]
            asset_indices.extend(asset_indices)
            portfolio_indices.extend([portf_index]*len(asset_indices))
            portf_index += 1
        asset_indices = torch.tensor(asset_indices, dtype=torch.int)
        portfolio_indices = torch.tensor(portfolio_indices, dtype=torch.int)
        scores = self(asset_indices, portfolio_indices, return_indvl_scores=return_indvl_scores, **fwd_kwargs)
        scores = self.asset_prior_evidence.unsqueeze(0) + scores
        if return_probs:
            scores = self.scores_to_prob(scores)
        if return_indvl_scores:
            # gather assets per portfolio
            portf_scores = []
            for i in range(portf_index):
                portf_score = scores[portfolio_indices==i]  #(num_assets_per_portfolio,num_assets)
                portf_scores.append(portf_score.tolist())
            if return_target_asset_keys:
                return portf_scores, self.target_asset_keys
            return portf_scores
        else:
            portf_scores = scores.tolist()
            if return_target_asset_keys:
                return portf_scores, self.target_asset_keys
            return portf_scores
        
######################################################################################
## EmbeddingRecommender
######################################################################################
class EmbeddingRecommender(BaseAssetRecommender):
    """
    Recommend Assets based on Embedding Similarities
    """
    def __init__(
            self,
            num_assets,
            emb_dim,
            asset_key_index_map,
            asset_counts,
            probs_norm_method = 'softmax',
            r_mask = 1,
            masking_strategy = 'random',
            feat_embedding_max_norm = None,
            target_embedding_max_norm = None,
            scale_emb_grad_by_freq = False,
            sparse_embedding = False,
            portf_emb_aggr_method = 'L2Norm',
            norm_asset_feat_embedding = False,
            use_clip_scores = False):
        super().__init__(num_assets=num_assets, asset_key_index_map=asset_key_index_map, probs_norm_method=probs_norm_method, r_mask=r_mask, masking_strategy=masking_strategy, asset_counts=asset_counts)
        self.emb_dim = emb_dim
        self.feat_embedding = nn.Embedding(num_embeddings=num_assets, embedding_dim=emb_dim, max_norm=feat_embedding_max_norm, scale_grad_by_freq=scale_emb_grad_by_freq, sparse=sparse_embedding)
        self.target_embedding = nn.Embedding(num_embeddings=num_assets, embedding_dim=emb_dim, max_norm=target_embedding_max_norm, scale_grad_by_freq=scale_emb_grad_by_freq)
        assert portf_emb_aggr_method in ('L2Norm', 'Mean', 'MLPWeightedSum', 'None', 'portf_emb_aggr_method must be one of (L2Norm, Mean, MLPWeightedSum, None)')
        self.portf_emb_aggr_method = portf_emb_aggr_method
        self.norm_asset_feat_embedding = norm_asset_feat_embedding
        self.use_clip_scores = use_clip_scores
        if self.use_clip_scores:
            # use two more bias parameters for pos and neg clipping of scores
            self.clip_bias = nn.Parameter(torch.zeros(2))
        if portf_emb_aggr_method == 'MLPWeightedSum':
            assert False, 'Not implemented yet'
            hidden_dim = 64
            dropout_rate = 0.2
            self.mlp = nn.Sequential(
                nn.Linear(num_assets,hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim,hidden_dim))  #todo

    def forward_feat_emb(self, asset_indices, portfolio_indices, return_indvl_scores = False):
        asset_feat_emb = self.feat_embedding(asset_indices)
        if self.norm_asset_feat_embedding:
            asset_feat_emb = asset_feat_emb/asset_feat_emb.norm(p=2, dim=-1, keepdim=True)
        # combine embeddings per portfolio to reduce computation (as this is mathematically the same as first doing the dot product and then summing up)
        portf_feat_emb = torch.zeros(portfolio_indices.max()+1, asset_feat_emb.shape[1]).to(asset_feat_emb.device)
        if self.portf_emb_aggr_method == 'Mean':
            reduce = 'mean'
        elif self.portf_emb_aggr_method == 'MLPWeightedSum':
            alpha = self.mlp(asset_feat_emb)
        else:
            reduce = 'sum'
        portf_feat_emb = portf_feat_emb.scatter_reduce(dim=0, index=portfolio_indices.unsqueeze(1).expand(-1,asset_feat_emb.shape[1]), src=asset_feat_emb, reduce=reduce, include_self=False)  #(B,D)
        if return_indvl_scores:
            # no consolidiation per portfolio, and no normalization, but scores should still be normalized by the norm of the resulting portfolio embedding (sum up)
            portf_norm = portf_feat_emb.norm(p=2, dim=-1)
            # repeat again per asset
            asset_portf_norm = torch.gather(portf_norm, dim=0, index=portfolio_indices)
            asset_feat_emb = asset_feat_emb / asset_portf_norm.unsqueeze(1)
            return asset_feat_emb
        if self.portf_emb_aggr_method == 'L2Norm':
            portf_feat_emb = portf_feat_emb/portf_feat_emb.norm(p=2, dim=-1, keepdim=True)
        
        return portf_feat_emb

    def forward(self, asset_indices, portfolio_indices, return_indvl_scores=False):
        """
        asset_indices: 
            int64 tensor of shape (B*num_assets_per_portfolio). 
            Contains indices of assets. B=number of portfolios
        portfolio_indices: 
            int64 tensor of shape (B*num_assets_per_portfolio). 
            Contains batch indices, i.e. which assets belong to which portfolio (has to start with index 0 and be consecutive)
        """
        feat_emb = self.forward_feat_emb(asset_indices, portfolio_indices, return_indvl_scores)            
        asset_targ_emb = self.target_embedding(torch.arange(self.num_assets).to(feat_emb.device))  #(num_assets,D)
        scores = torch.matmul(feat_emb, asset_targ_emb.T)  #(B,num_assets) or (B*num_assets_per_portfolio,num_assets)
        return scores

######################################################################################
## MLP Model
    # todo. Also include bottleneck layer for visualisation purposes, with options to 
    # use normal projection layer at the end (=dot product) or inverse euclidean 
    # distance metric (for more intuitive clustering). Can later be extended with e.g.
    # gaussian mixture model for clustering
######################################################################################
class MLPRecommender(BaseAssetRecommender):
    def __init__(
            self,
            num_assets,
            hidden_dim,
            num_hidden_layers,
            asset_key_index_map,
            asset_counts,
            probs_norm_method = 'softmax',
            r_mask = 1,
            masking_strategy = 'random',
            dropout_rate=0.0, 
            bottleneck_dim=None,
            norm_portf_emb_by_portf_size=False):
        super().__init__(num_assets=num_assets, asset_key_index_map=asset_key_index_map, probs_norm_method=probs_norm_method, r_mask=r_mask, masking_strategy=masking_strategy, asset_counts=asset_counts)
        self.num_assets = num_assets
        #self.dropout_rate = dropout_rate  
        self.mlp_hidden_dim = hidden_dim
        self.feat_embedding = nn.Embedding(num_embeddings=num_assets, embedding_dim=hidden_dim)
        self.mlp_hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.mlp_hidden_layers.append(
                nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(hidden_dim,hidden_dim),  # Eingabe: zweifache Dimension der Embeddings (feat und target)
                    nn.ReLU()))
        last_dim = hidden_dim
        if bottleneck_dim is not None:
            self.mlp_hidden_layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            last_dim = bottleneck_dim
        self.pred_layer = nn.Sequential(
            nn.Linear(last_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_assets))
        self.norm_portf_emb_by_portf_size = norm_portf_emb_by_portf_size

    def forward_to_final_layer(self, asset_indices, portfolio_indices):
        # portf_feat = F.one_hot(asset_indices, num_classes = self.num_assets).float()
        # x = torch.zeros(portfolio_indices.max()+1, self.num_assets).to(asset_indices.device)
        # x = x.scatter_reduce(dim=0, index=portfolio_indices.unsqueeze(1).expand(-1,portf_feat.shape[1]), src=portf_feat, reduce='sum')  #(B,D)
        asset_feat_emb = self.feat_embedding(asset_indices)
        x = torch.zeros(portfolio_indices.max()+1, asset_feat_emb.shape[1]).to(asset_feat_emb.device)
        if self.norm_portf_emb_by_portf_size:
            reduce = 'mean'  # divide score by number of portfolios, i.e. take mean of individual
        else:
            reduce = 'sum'   # use sum
        x = x.scatter_reduce(dim=0, index=portfolio_indices.unsqueeze(1).expand(-1,asset_feat_emb.shape[1]), src=asset_feat_emb, reduce=reduce, include_self=False)  #(B,D)
        for layer in self.mlp_hidden_layers:
            x = layer(x)
        return x

    def forward(self, asset_indices, portfolio_indices, **kwargs):
        """
        asset_indices: 
            int64 tensor of shape (B*num_assets_per_portfolio). 
            Contains indices of assets. B=number of portfolios
        portfolio_indices: 
            int64 tensor of shape (B*num_assets_per_portfolio). 
            Contains batch indices, i.e. which assets belong to which portfolio
        """
        x = self.forward_to_final_layer(asset_indices, portfolio_indices)
        x = self.pred_layer(x)
        return x


######################################################################################
## Lightning Wrapper
######################################################################################
class LitRecommenderWrapper(L.LightningModule):
    """Model class"""
    def __init__(self, model_name, model_config, ign_self_scores=True, eval_train_every_n_epoch=10):
        super().__init__()
        if model_name == 'EmbeddingRecommender':
            self.recommender = EmbeddingRecommender(**model_config)
        elif model_name == 'MLPRecommender':
            self.recommender = MLPRecommender(**model_config)
        else:
            assert False, f'Unknown Model Name "{model_name}"'
        if isinstance(self.recommender.scores_to_prob, nn.Softmax):
            self.loss = F.cross_entropy
            self.task = 'multiclass'
        elif isinstance(self.recommender.scores_to_prob, nn.Sigmoid):
            self.loss = lambda input, target: torch.mean(torch.sum((target+(1-target)/target.shape[1])*(F.binary_cross_entropy_with_logits(input,target, reduction='none')), dim=1))  #negative values are lower weighted (by factor 1/num_assets)
            self.task = 'multilabel'
        self.ign_self_scores = ign_self_scores
        top_k_75perc = int(self.recommender.num_assets*0.25)
        top_k_95perc = int(self.recommender.num_assets*0.05)
        if self.task == 'multiclass':
            self.train_auroc = AUROC(task=self.task, num_classes=self.recommender.num_assets, average='macro', thresholds = 100)
            self.valid_auroc = nn.ModuleList([AUROC(task=self.task, num_classes=self.recommender.num_assets, average='macro', thresholds = 100) for i in range(2)])
            self.train_accuracy_1 = Accuracy(task=self.task, num_classes=self.recommender.num_assets, average='macro', top_k=top_k_75perc)
            self.train_accuracy_2 = Accuracy(task=self.task, num_classes=self.recommender.num_assets, average='macro', top_k=top_k_95perc)
            self.valid_accuracy_1 = nn.ModuleList([Accuracy(task=self.task, num_classes=self.recommender.num_assets, average='macro', top_k=top_k_75perc) for i in range(2)])
            self.valid_accuracy_2 = nn.ModuleList([Accuracy(task=self.task, num_classes=self.recommender.num_assets, average='macro', top_k=top_k_95perc) for i in range(2)])
            self.acc_label = [' top 75%', ' top 95%']
        else:
            self.train_auroc = AUROC(task=self.task, num_labels=self.recommender.num_assets, average='macro', thresholds = 100)
            self.valid_auroc = nn.ModuleList([AUROC(task=self.task, num_labels=self.recommender.num_assets, average='macro', thresholds = 100) for i in range(2)])
            self.train_accuracy_1 = Accuracy(task=self.task, num_labels=self.recommender.num_assets, average='macro', threshold=0.5)
            self.train_accuracy_2 = Accuracy(task=self.task, num_labels=self.recommender.num_assets, average='macro', threshold=0.75)
            self.valid_accuracy_1 = nn.ModuleList([Accuracy(task=self.task, num_labels=self.recommender.num_assets, average='macro', threshold=0.5) for i in range(2)])
            self.valid_accuracy_2 = nn.ModuleList([Accuracy(task=self.task, num_labels=self.recommender.num_assets, average='macro', threshold=0.75) for i in range(2)])
            self.acc_label = [' 0.5 threshold', ' 0.75 threshold']
        self.train_average_precision = MultilabelRankingAveragePrecision(num_labels=self.recommender.num_assets)
        self.valid_average_precision = nn.ModuleList([MultilabelRankingAveragePrecision(num_labels=self.recommender.num_assets) for i in range(2)])

        self.save_hyperparameters()
        self.epoch_counter = 1
        self.eval_train_every_n_epoch = eval_train_every_n_epoch
        
    def training_step(self, batch, batch_idx):
        portfolio_indices, asset_indices, add_features, train_mask, _ = batch
        portfolio_indices = torch.masked_select(portfolio_indices, train_mask)
        asset_indices = torch.masked_select(asset_indices, train_mask)
        logits, target = self.recommender.forward_masking(asset_indices, portfolio_indices, ign_self_scores=self.ign_self_scores)
        loss = self.loss(input=logits, target=target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if self.epoch_counter%self.eval_train_every_n_epoch == 0:
            target = target.long()
            self.train_average_precision(logits, target)
            if self.task == 'multiclass':
                target = target.argmax(dim=1)
                self.train_auroc(logits, target)
            self.train_accuracy_1(logits, target)
            self.train_accuracy_2(logits, target)
        return loss
    
    def on_train_epoch_end(self):
        if self.epoch_counter%self.eval_train_every_n_epoch == 0:
            # log epoch metric
            self.log('train_auroc', self.train_auroc)
            self.log('train_accuracy '+self.acc_label[0], self.train_accuracy_1)
            self.log('train_accuracy '+self.acc_label[1], self.train_accuracy_2)
            self.log('train_average_precision', self.train_average_precision)
        self.epoch_counter += 1
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
            # validate on masked out portfolios
            portfolio_indices, asset_indices, add_features, train_mask, val_mask = batch
            if dataloader_idx == 0:
                logits, target = self.recommender.forward_masking(asset_indices, portfolio_indices, ign_self_scores=self.ign_self_scores)   #todo: should be with masking_strategy = all_combinations
                loss = self.loss(input=logits, target=target)
                self.log('val_loss_masked_portf', loss, on_step=False, on_epoch=True, prog_bar=True)
                target = target.long().ceil()  # round back up to 1 for each target
                self.valid_average_precision[dataloader_idx](logits, target)
                if self.task == 'multiclass':
                    target = target.argmax(dim=1)
                self.valid_auroc[dataloader_idx](logits, target)
                self.valid_accuracy_1[dataloader_idx](logits, target)
                self.valid_accuracy_2[dataloader_idx](logits, target)

            # validate on masked out assets (edges) (both dataloaders)
            elif dataloader_idx == 1:
                target = torch.zeros(portfolio_indices.max()+1, self.recommender.num_assets).to(asset_indices.device)
                target[torch.masked_select(portfolio_indices, val_mask), torch.masked_select(asset_indices, val_mask)] = 1.
                target = target / target.sum(1, keepdim=True)  #so that the val_loss is comparable to the train_loss
                portfolio_indices = torch.masked_select(portfolio_indices, train_mask)
                asset_indices = torch.masked_select(asset_indices, train_mask)
                logits = self.recommender(asset_indices, portfolio_indices, return_indvl_scores=False)
                logits = logits + self.recommender.asset_prior_evidence.unsqueeze(0)
                if self.ign_self_scores:
                    scores_log_adj = torch.zeros_like(logits)
                    scores_log_adj[portfolio_indices, asset_indices] = -1e30
                    logits = logits + scores_log_adj
                # only keep the rows where at least one position is included in the target
                idx = torch.where(target.sum(1) > 0)[0]
                logits = logits[idx]
                target = target[idx]
                loss = self.loss(input=logits, target=target)
                self.log('val_loss_masked_asset', loss, on_step=False, on_epoch=True, prog_bar=True)
                target = target.long().ceil()  # round back up to 1 for each target
                self.valid_average_precision[dataloader_idx](logits, target)
                if self.task == 'multiclass':
                    target = target.argmax(dim=1)
                self.valid_auroc[dataloader_idx](logits, target)
                self.valid_accuracy_1[dataloader_idx](logits, target)
                self.valid_accuracy_2[dataloader_idx](logits, target)
        
        
    def on_validation_epoch_end(self):
        # log epoch metric
        postfix = ['masked_portf', 'masked_asset']
        for i in range(2):
            self.log('valid_auroc ' + postfix[i], self.valid_auroc[i])
            self.log('valid_accuracy ' + postfix[i] + self.acc_label[0], self.valid_accuracy_1[i])
            self.log('valid_accuracy ' + postfix[i] + self.acc_label[1], self.valid_accuracy_2[i])
            self.log('valid_average_precision ' + postfix[i], self.valid_average_precision[i])
        
    def configure_optimizers(self):
        if hasattr(self.recommender, 'feat_embedding') and self.recommender.feat_embedding.sparse == True:
            # here we need two optimizers to handle the sparse embedding
            parameters_nonsparse = []
            for name, module in self.recommender.named_children():
                if name != 'feat_embedding':
                    parameters_nonsparse.extend(list(module.parameters()))
            return torch.optim.AdamW(parameters_nonsparse), torch.optim.SparseAdam(self.recommender.feat_embedding.parameters())
        return torch.optim.AdamW(self.parameters())
    
    #todo: implement a final eval method at the end of training, where the validation metrics are calculated for each frequency bin of the masked assets (use dataset.bucket_assignments)
    

    
    
    
    
