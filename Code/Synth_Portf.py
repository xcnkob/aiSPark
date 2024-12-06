# -*- coding: utf-8 -*-
"""
to do for cient preferences task: 
1. per client, sample masking rate (between 0 and 1, or mabe better sometimes 0, and sometimes closer to 1) of assets
2. calculate aggregated portfolio features using masked assets only, and target aggregated values using all assets
3. Predict target aggregated features using all features (use cross entropy, so that we are in the same space)
Including cases where we only use person features assures that these correlations are also taken into account. Maybe it would also work
if we use a sum model, so that an initial base of the prediction comes from the person features, and it is refined using the (aggregated?) 
asset feaures

If in adition to use cross entropy loss for asset embedding, we are calculating the probabilities of each asset feature of the target asset
by aggregating the features of each asset with their predicted probability and use cross entropy loss with the target asset features, 
we further align the embeddings (client and asset) according to the asset features (so we can directly get e.g. sector affinities, country affinities, etc.)
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from VAE_Models import LitVAE
# from VAE_Models import GaussianVAE, CategoricalVAE, PoissonVAE
import lightning as L
import math
import ast
from pathlib import Path

# class LitVAE(L.LightningModule):
#     def __init__(
#         self,
#         model_name,
#         model_config,
#         learning_rate: float = 1e-3,
#         initial_temperature: float = 1.0,
#         min_temperature: float = 0.1,
#         temperature_decay_rate: float = 0.05,
#         initial_beta_kl: float = 0.01,
#         beta_kl_max: float = 1 ,
#         beta_kl_growth_rate: float = 0.2
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         self.learning_rate = learning_rate
#         # Store temperature parameters
#         self.initial_temperature = initial_temperature
#         self.min_temperature = min_temperature
#         self.temperature_decay_rate = temperature_decay_rate
#         # Current temperature (will be updated during training)
#         self.current_temperature = initial_temperature
#         if 'temperature' not in model_config and model_name != 'GaussianVAE':
#             model_config['temperature'] = self.initial_temperature

#         # Store beta_kl
#         self.initial_beta_kl = initial_beta_kl
#         self.beta_kl_max = beta_kl_max
#         self.beta_kl_growth_rate = beta_kl_growth_rate
#         self.current_beta_kl = self.initial_beta_kl
#         if 'beta_kl' not in model_config:
#             model_config['beta_kl'] = self.initial_beta_kl

#         if model_name == 'GaussianVAE':
#             self.vae = GaussianVAE(**model_config)
#         elif model_name == 'CategoricalVAE':
#             self.vae = CategoricalVAE(**model_config)
#         elif model_name == 'PoissonVAE':
#             self.vae = PoissonVAE(**model_config)
#         else:
#             assert False, 'model_name must be one of (GaussianVAE, CategoricalVAE, PoissonVAE)'

#     def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
#         portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch

#         # Update VAE's temperature if it's a CategoricalVAE
#         if isinstance(self.vae, CategoricalVAE) or isinstance(self.vae, PoissonVAE):
#             self.vae.set_temperature(self.current_temperature)

#         # Update beta_kl
#         self.vae.set_beta_kl(self.current_beta_kl)

#         x_recon, distribution_params = self.vae(client_features)
#         losses = self.vae.loss_function(client_features, x_recon, *distribution_params)

#         # Log temperature
#         self.log('temperature', self.current_temperature, on_step=False, on_epoch=True, batch_size=client_features.shape[0])
#         # Log beta_kl
#         self.log('beta_kl', self.current_beta_kl, on_step=False, on_epoch=True, batch_size=client_features.shape[0])

#         # Log all loss components
#         for name, value in losses.items():
#             if name=='total_loss':
#                 self.log(f'train_{name}', value, on_step=True, on_epoch=True, prog_bar=True, batch_size=client_features.shape[0])
#             else:
#                 self.log(f'train_{name}', value, on_step=False, on_epoch=True, prog_bar=False, batch_size=client_features.shape[0])
#         return losses['total_loss']
    
#     def on_train_epoch_end(self):
#         # Anneal temperature using exponential decay
#         self.current_temperature = max(
#             self.min_temperature, 
#             self.initial_temperature * math.exp(-self.temperature_decay_rate * self.current_epoch)
#         )
#         # Anneal beta_kl using exponential growth
#         self.current_beta_kl = min(
#             self.beta_kl_max, 
#             self.initial_beta_kl * math.exp(self.beta_kl_growth_rate * self.current_epoch)
#         )
    
#     def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
#         portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch
#         x_recon, distribution_params = self.vae(client_features)
#         losses = self.vae.loss_function(client_features, x_recon, *distribution_params)
#         # Log all loss components
#         for name, value in losses.items():
#             if name=='total_loss':
#                 self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=True, batch_size=client_features.shape[0])
#             else:
#                 self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=False, batch_size=client_features.shape[0])

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

"""
Dataset Class for Synthetic Portfolio Dataset
"""
class SynthPortfDataset(Dataset):
    def __init__(self, dataset_parent_path, edge_mask_ratio=0.2, num_asset_freq_buckets=10, aggr_portf_feat_using_allocation_percentage=True, apply_mask_for_aggr=False):
        self.dataset_parent_path = dataset_parent_path
        self.clients_df = pd.read_csv(self.dataset_parent_path + 'clients.csv')
        self.assets_df = pd.read_csv(self.dataset_parent_path + 'assets.csv')
        self.portfolios_df = pd.read_csv(self.dataset_parent_path + 'portfolios.csv')
        self.num_asset_freq_buckets = num_asset_freq_buckets
        self.edge_mask_ratio = edge_mask_ratio
        self.aggr_portf_feat_using_allocation_percentage = aggr_portf_feat_using_allocation_percentage
        self.apply_mask_for_aggr = apply_mask_for_aggr
        print('Start preprocessing Dataset')
        self._preprocess()
        print(f'Init Dataset End. Number of unique assets: {self.assets_df['asset_id'].count()}. Number of unique portfolios: {self.num_portfolios}')
        
    def __len__(self):
        return self.num_portfolios
    
    def __getitems__(self, indices):
        clients = self.clients_df.iloc[indices]
        portf_sel_idx = self.portfolios_df['client_id'].isin(clients['client_id'])
        portfolios = self.portfolios_df.loc[portf_sel_idx]
        portfolio_indices = torch.tensor(pd.factorize(portfolios['client_id'])[0])  #reset indices starting from 0 for the current batch
        asset_indices = torch.tensor(portfolios['asset_id'].values)
        
        client_features = torch.tensor(clients.iloc[:,2:].astype('float32').values)  #ignore unnamed column and client_id
        client_target_latents = torch.tensor(self.client_latents_df.iloc[indices,2:].astype('float32').values)
        portfolio_mask = self.portfolio_val_mask[portf_sel_idx.values]
        portfolio_val_mask = portfolio_mask==1
        portfolio_train_mask = portfolio_mask==0
        return portfolio_indices, asset_indices, client_features, portfolio_train_mask, portfolio_val_mask, client_target_latents
    
    def convert_text_to_integers(self, df):
        column_maps = {}
        for column in df.columns:
            if column in ('asset_id', 'client_id'):
                # Replace IDs with consecutive integers starting from 0
                df[column], codes = pd.factorize(df[column], sort=True)
                column_maps[column] = dict(zip(codes, df[column].unique()))
            elif df[column].dtype == 'object':
                # Check if the column contains lists
                if isinstance(df[column].iloc[0], list):
                    # Flatten the list to factorize unique values across all rows
                    all_values = [item for sublist in df[column] for item in sublist]
                    
                    # Create factorized mapping for list values
                    unique_values = np.unique(all_values)
                    codes = np.arange(len(unique_values))
                    column_maps[column] = dict(zip(unique_values, codes))
                    
                    # Convert lists to integer codes
                    df[column] = df[column].apply(lambda x: [column_maps[column][val] for val in x])
                else:
                    # Handle regular text columns
                    df[column], codes = pd.factorize(df[column])
                    column_maps[column] = dict(zip(codes, df[column].unique()))
        
        return column_maps
    
    def one_hot_encode(self, df, columns, value_map):
        ignore_columns = []
        for column in columns:
            if isinstance(df[column].iloc[0], list):
                # Create a DataFrame of binary columns
                factorized_columns = pd.DataFrame(
                    {val: df[column].apply(lambda x: True if val in x else False) 
                    for _, val in value_map[column].items()},
                    index=df.index
                )
                # Rename columns to avoid potential conflicts
                factorized_columns.columns = [f"{column}_{val}" for _, val in value_map[column].items()]
                
                # Drop the original column and concatenate the new columns
                df = pd.concat([df.drop(columns=[column]), factorized_columns], axis=1)
                ignore_columns.append(column)
        columns = [column for column in columns if not column in ignore_columns]
        # for non-list columns, simply use pd.dummies
        return pd.get_dummies(df, columns=columns)
    
    def _preprocess(self):
        # parse list column to real list instead of string
        self.clients_df['preferred_sectors'] = self.clients_df['preferred_sectors'].apply(ast.literal_eval)
        ### Create train and validation masks ###
        self.asset_counts = self.portfolios_df.groupby('asset_id')['client_id'].count()
        self.num_portfolios = self.portfolios_df['client_id'].nunique()

        # Calculate asset frequencies (in terms of the percentage of portfolios they occur in)
        asset_counts = torch.tensor(self.asset_counts.values)
        self.asset_frequencies = asset_counts / self.num_portfolios

        # Create asset frequency buckets
        bucket_counts, bucket_upper_edges = torch.histogram(self.asset_frequencies, self.num_asset_freq_buckets)
        self.bucket_upper_edges = bucket_upper_edges[1:]
        self.bucket_assignments = torch.bucketize(self.asset_frequencies, self.bucket_upper_edges)

        # Create stratified mask for validation
        self.portfolio_val_mask = torch.zeros(self.portfolios_df['client_id'].count())
        asset_indices = torch.arange(self.portfolios_df['asset_id'].count())
        for i, bucket_size in enumerate(bucket_counts):
            if bucket_size == 0:
                continue
            # select edge_mask_ratio of edges of assets in the current bucket and remove them (=add to val_mask)
            relv_assets_indices = torch.where(self.bucket_assignments==i)[0]
            relv_edge_indices = torch.where(torch.isin(asset_indices, relv_assets_indices))[0]
            num_masked = int(relv_edge_indices.shape[0] * self.edge_mask_ratio)
            masked_edge_indices = np.random.choice(relv_edge_indices, num_masked, replace=False)
            self.portfolio_val_mask[masked_edge_indices] = 1

        ### Preprocessing of data ###
        self.assets_df.rename(columns={'country': 'asset_country'}, inplace=True)
        # Split client dataframe into features and target latents (latter are only used for validation; for real dataset, we don't know these attributes)
        self.client_latents_df = pd.concat([self.clients_df.iloc[:,:2], self.clients_df.iloc[:,6:]], axis=1)
        self.clients_df = self.clients_df.iloc[:,:6]

        # Replace categorical values with indices and store maps
        self.asset_feat_map = self.convert_text_to_integers(self.assets_df)
        self.client_feat_map = self.convert_text_to_integers(self.clients_df)
        self.client_latents_map = self.convert_text_to_integers(self.client_latents_df)
        self.portfolio_feat_map = self.convert_text_to_integers(self.portfolios_df)
        
        # Use aggregated portfolio features as additional client features
        # First one-hot encode categorical values
        asset_feature_columns = [key for key in self.asset_feat_map.keys() if key not in ('asset_id')]
        client_feature_columns = [key for key in self.client_feat_map.keys() if key not in ('client_id')]
        client_latents_columns = [key for key in self.client_latents_map.keys() if key not in ('client_id')]

        asset_df_one_hot = self.one_hot_encode(self.assets_df, columns=asset_feature_columns, value_map=self.asset_feat_map)
        self.clients_df = self.one_hot_encode(self.clients_df, columns=client_feature_columns, value_map=self.client_feat_map)
        self.client_latents_df = self.one_hot_encode(self.client_latents_df, columns=client_latents_columns, value_map=self.client_latents_map)

        # Then do (weighted) aggregation of asset features per portfolio
        aggr_feature_cols = [col for col in asset_df_one_hot.columns[2:]]

        def weighted_aggr(df):
            if self.aggr_portf_feat_using_allocation_percentage:
                weights = df['allocation_percentage']
                agg_dict = {
                    f'{col}': (df[col] * weights).sum() / weights.sum()
                    for col in aggr_feature_cols
                }
            else:
                agg_dict = {
                    f'{col}': (df[col] * weights).mean()
                    for col in aggr_feature_cols
                }
            agg_dict['portfolio_value'] = df['allocation_amount'].sum()
            return pd.Series(agg_dict)

        # Group by client_id and perform weighted or simple aggregation
        merged_df = self.portfolios_df.merge(asset_df_one_hot, on='asset_id')

        if self.apply_mask_for_aggr:
            merged_df = merged_df[self.train_mask]
        aggr_portfolio_features = merged_df.groupby('client_id').apply(weighted_aggr).reset_index()
        self.clients_df = self.clients_df.merge(aggr_portfolio_features, on='client_id')
        
        # define list of ddics containing feature name, type and section size (correct order is important)
        # feature types: 
        # - categorical: used for one-hot categorical values
        # - distribution: used for general probility distribution, i.e. several values summing up to 1
        # - score: used for a single percentage score between 0 and 1
        # - mumerical: used for general numerical value [-inf, +inf]
        
        # client features
        self.client_features_descn = [{'feature_name': key, 'feature_type': 'categorical', 'feature_size': len(value)} for key, value in self.client_feat_map.items() if key not in ('client_id')]
        self.client_features_descn.extend({'feature_name': name, 'feature_type': 'score', 'feature_size': 1} for name in ['dividend_yield', 'volatility_risk', 'esg_score'])
        self.client_features_descn.extend([{'feature_name': key, 'feature_type': 'distribution', 'feature_size': len(value)} for key, value in self.asset_feat_map.items() if key not in ('asset_id')])
        self.client_features_descn.append({'feature_name': 'portfolio_value', 'feature_type': 'numerical', 'feature_size': 1})
        # client latents
        self.client_latents_descn = [{'feature_name': name, 'feature_type': 'score', 'feature_size': 1} for name in ['tech_affinity', 'international_exposure', 'risk_tolerance', 'esg_preference', 'dividend_preference','growth_preference', 'small_cap_affinity', 'home_bias']]
        self.client_latents_descn.extend([{'feature_name': name, 'feature_type': 'multiclass' if name == 'preferred_sectors' else 'categorical', 'feature_size': len(self.client_latents_map[name])} for name in ['preferred_sectors', 'investment_style', 'diversity_level']])

        def scale_features(df, feature_descn_list):
            for feat_descn in feature_descn_list:
                if feat_descn['feature_type'] == 'score':
                    # scale between 0 and 1
                    df[feat_descn['feature_name']] = (df[feat_descn['feature_name']] - df[feat_descn['feature_name']].min()) / (df[feat_descn['feature_name']].max() - df[feat_descn['feature_name']].min())
                elif feat_descn['feature_type'] == 'numerical':
                    # Normalize (0 mean and 1 std)
                    df[feat_descn['feature_name']] = (df[feat_descn['feature_name']] - df[feat_descn['feature_name']].mean()) / df[feat_descn['feature_name']].std()

        scale_features(self.clients_df, self.client_features_descn)
        scale_features(self.client_latents_df, self.client_latents_descn)

        
        # Get split sections of different types of features
        def get_split_sections(feature_descn_list):
            split_sections = []
            feature_type = []
            prev_feat_type = None
            section_size = 0
            for feat_descn in feature_descn_list:
                feat_type = feat_descn['feature_type']
                if feat_type != prev_feat_type and prev_feat_type is not None:
                    # first add size of accumalated features so far (if any)
                    split_sections.append(section_size)
                    feature_type.append(prev_feat_type)
                    #reset
                    prev_feat_type = None
                    section_size = 0
                if feat_type in ('categorical', 'distribution', 'multiclass'):
                    # always split directly
                    split_sections.append(feat_descn['feature_size'])
                    feature_type.append(feat_type)
                else:
                    # start aggregating
                    section_size += feat_descn['feature_size']
                    prev_feat_type = feat_type
            # add remainder (if any)
            if prev_feat_type is not None:
                split_sections.append(section_size)
                feature_type.append(prev_feat_type)

            return split_sections, feature_type

        self.client_feat_split_sections, self.client_feat_types = get_split_sections(self.client_features_descn)
        self.client_latents_split_sections, self.client_latents_types = get_split_sections(self.client_latents_descn)        
    
def collate_fn(batch):
    return batch

def create_data_loaders(dataset, batch_size, val_split_ratio=0.1, test_split_ratio=0.1, num_workers=0):
    # Split dataset into training and validation. 
    # For now use random split. todo: Would be better to split taking into account the number of assets per portfolio
    torch.manual_seed(0)   # fix for reproducability
    val_size = int(val_split_ratio * len(dataset))
    test_size = int(test_split_ratio * len(dataset))
    train_size = len(dataset)-val_size-test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=num_workers, persistent_workers=True if num_workers > 0 else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def plot_dataset(dataset):
    import matplotlib.pyplot as plt
        
    asset_group = dataset.portfolios_df.groupby('asset_id')['client_id'].count()
    portfolio_group = dataset.portfolios_df.groupby('client_id')['asset_id'].count()
    x = []
    asset_counts = []
    portfolio_counts = []
    for i in range(max(asset_group.max(), portfolio_group.max())):
        x.append(i)
        asset_counts.append(asset_group[asset_group==i].count())
        portfolio_counts.append(portfolio_group[portfolio_group==i].count())
    f, ax = plt.subplots(2)
    color = 'tab:red'
    ln1 = ax[0].plot(x, asset_counts, label='asset occurences', color=color)
    ax[0].set_xlabel('asset occurence / portfolio size')
    ax[0].set_ylabel('asset counts', color=color)
    ax[0].tick_params(axis='y', labelcolor=color)
    ax2 = ax[0].twinx()
    color = 'tab:blue'
    ln2 = ax2.plot(x, portfolio_counts, label= 'portfolio size', color=color)
    ax2.set_ylabel('portfolio counts', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax[0].legend(lns, labs, loc=0)
    plt.title('Marginal occurences')

    color = 'tab:red'
    ln1 = ax[1].plot(x, [sum(asset_counts[i:]) for i in x], label='included assets', color=color)
    ax[1].set_xlabel('asset occurence threshold / portfolio size threshold')
    ax[1].set_ylabel('Number of included assets', color=color)
    ax[1].tick_params(axis='y', labelcolor=color)
    ax2 = ax[1].twinx()
    color = 'tab:blue'
    ln2 = ax2.plot(x, [sum(portfolio_counts[i:]) for i in x], label= 'included portfolios', color=color)
    ax2.set_ylabel('Number of included portfolios', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=0)
    plt.title('Cumulative occurences')
    f.tight_layout()
    f.savefig('dataset_asset_portfolio_counts_and_threshold.png')
    # plt.show()

def plot_embeddings(save_dir, client_emb, target_latents, target_latents_labels, client_latents_map):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # todo: handle last column with list value
    target_latents_labels = [x for x in target_latents_labels if x not in ('client_id', 'preferred_sectors')]

    if client_emb.shape[1] > 2:
        print('Start T-SNE embeddings')
        client_emb = TSNE(n_components=2, learning_rate='auto', perplexity=30).fit_transform(client_emb.numpy())
        print('End T-SNE embeddings')
    
    
    f, ax = plt.subplots(1)
    ax.scatter(client_emb[:, 0], client_emb[:, 1], 
                s=1,
                alpha=0.7)
    ax.set_title('Client Embeddings')
    f.tight_layout()
    f.savefig(save_dir+'/Client_Embeddings.png')

    for i,target_label in enumerate(target_latents_labels):
        f, ax = plt.subplots(1)
        if target_label in client_latents_map.keys():
            # categories
            unique_categories = client_latents_map[target_label]
            category_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_categories)))
            for j, category in enumerate(unique_categories):
                mask = target_latents[:,j] == unique_categories[category]
                ax.scatter(client_emb[mask, 0], client_emb[mask, 1], 
                            color=category_colors[j], 
                            label=category,
                            s=2,
                            alpha=0.7)
            ax.legend()
        else:
            scatter = ax.scatter(client_emb[:, 0], client_emb[:, 1], c=target_latents[:,i], alpha=0.7, s=2, cmap='viridis')
            plt.colorbar(scatter, ax=ax, label=target_label)

        ax.set_title('Client Embeddings: ' + target_label)
        f.tight_layout()
        f.savefig(save_dir+'/Client_Embeddings_'+target_label+'.png')


"""
Class to train Linear Head for evaluation of prediction of client latents based on model embeddings
"""
class EmbeddingEvaluator(L.LightningModule):
    def __init__(
        self,
        base_model: L.LightningModule,
        output_dim: int,
        target_split_sections: list[int], 
        split_section_types: list[str], 
        target_feature_names: list[str],
        loss_map: dict,
        learning_rate: float = 1e-3
    ):
        super().__init__()

        assert len(target_feature_names) == len(target_split_sections), 'len(target_feature_names) must be same as len(target_split_sections)'

        self.base_model = base_model
        # Freeze VAE parameters
        for param in self.base_model.vae.parameters():
            param.requires_grad = False
        
        # Get embedding dimension and number of classes
        input_dim = self.base_model.vae.get_decoded_latent_dim()
        
        # Prepare linear head
        self.linear_head = nn.Linear(input_dim, output_dim).to(self.device)

        # Override output_split_sections and split_section_types to fit new target values
        self.base_model.vae.output_split_sections = target_split_sections
        self.base_model.vae.split_section_types = split_section_types
        self.base_model.vae.parse_loss_map(loss_map)

        self.target_feature_names = target_feature_names
        self.learning_rate = learning_rate

        # Save Hyperparameters
        self.hparams['model_type'] = self.base_model.hparams.model_name
        self.hparams['embed_dim'] = input_dim

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch
        
        # Extract embeddings (detach to stop gradient)
        embeddings = self.base_model.vae.extract_embeddings(client_features).detach()
        outputs = self.linear_head(embeddings)
        loss = self.base_model.vae.recon_loss_sections(outputs, client_target_latents)

        self.log(f'embed_eval_train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=embeddings.shape[0])

        return loss
    
    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        portfolio_indices, asset_indices, client_features, train_mask, val_mask, client_target_latents = batch
        embeddings = self.base_model.vae.extract_embeddings(client_features.to(self.device))
        outputs = self.linear_head(embeddings)
        section_losses = torch.stack(self.base_model.vae.recon_loss_sections(outputs, client_target_latents.to(self.device), return_indvl_losses=True))
        
        for i,target_feature_name in enumerate(self.target_feature_names):
            self.log(f'embed_eval_val_loss_'+target_feature_name, section_losses[i], prog_bar=False, on_step=False, on_epoch=True, batch_size=embeddings.shape[0])
        self.log(f'embed_eval_val_loss_mean', section_losses.sum(), prog_bar=True, on_step=False, on_epoch=True, batch_size=embeddings.shape[0])

    def configure_optimizers(self):
        return torch.optim.Adam(self.linear_head.parameters(), lr=self.learning_rate)


def eval_model(
    base_model: L.LightningModule, 
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    log_dir: str
):
    from pytorch_lightning.loggers import TensorBoardLogger

    # Do Model Evaluation on Client Latents
    loss_map = {'numerical': 'mse', 'categorical': 'cross_entropy', 'distribution': 'symmetric_kl', 'score': 'symmetric_kl', 'multiclass': 'binary_cross_entropy'}
    target_split_sections = []
    split_section_types = []
    target_feature_names = []
    for feat_descn in dataset.client_latents_descn:
        split_section_types.append(feat_descn['feature_type'])
        target_split_sections.append(feat_descn['feature_size'])
        target_feature_names.append(feat_descn['feature_name'])

    embedding_evaluator = EmbeddingEvaluator(
        base_model = base_model,
        output_dim = sum(target_split_sections),
        target_split_sections = target_split_sections, 
        split_section_types = split_section_types, 
        target_feature_names = target_feature_names,
        loss_map = loss_map,
        learning_rate = 1e-3)
    
    embedding_logger = TensorBoardLogger(
        save_dir=log_dir,
        name='embedding_evaluation',  # subfolder
        version=''  # prevents creating an additional version folder
    )

    # Create a new trainer with this logger
    embedding_trainer = L.Trainer(
        logger=embedding_logger,
        max_epochs=30,
        accelerator="gpu",
    )

    # Train the linear head and evaluate
    embedding_trainer.fit(
        embedding_evaluator, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == '__main__':

    from sklearn.model_selection import ParameterGrid

    train_model = False
    visualize_dataset = False

    dataset_parent_path = 'Datasets/Synth_Portf/Default_C10000_A500/'

    param_grid = [
    # {'model_name': ['GaussianVAE', 'PoissonVAE'],
    #  'latent_dim': [8,32,64,128],
    #  'enc_hidden_dims': [[], [64,64]],
    #  'dec_hidden_dims': [[], [64,64]],
    # },
    {'model_name': ['CategoricalVAE'], 
     'latent_dim': [32,128],
     'enc_hidden_dims': [[], [64,64]],
     'dec_hidden_dims': [[], [64,64]],
     'num_categories': [2,8]
    },
    {'model_name': ['CategoricalVAE'], 
     'latent_dim': [1],
     'enc_hidden_dims': [[], [64,64]],
     'dec_hidden_dims': [[], [64,64]],
     'num_categories': [64,128]
    },
    ]
    
    dataset = SynthPortfDataset(dataset_parent_path, edge_mask_ratio=0.2, num_asset_freq_buckets=10, aggr_portf_feat_using_allocation_percentage=True, apply_mask_for_aggr=False)

    if visualize_dataset:
        plot_dataset(dataset)

    train_loader, val_loader, _ = create_data_loaders(dataset=dataset, batch_size=256, val_split_ratio=0.1, test_split_ratio=0., num_workers=0)

    if train_model:
        ckpt_path = None
        for param in ParameterGrid(param_grid):
            model_name = param['model_name']
            if len(param['enc_hidden_dims']) == 0:
                encoder_name = 'Lin'
            else:
                encoder_name = 'MLP'
                for dim in param['enc_hidden_dims']:
                    encoder_name += '-' + str(dim)
            if len(param['dec_hidden_dims']) == 0:
                decoder_name = 'Lin'
            else:
                decoder_name = 'MLP'
                for dim in param['dec_hidden_dims']:
                    decoder_name += '-' + str(dim)
            
            cat_suffix = f'x{param['num_categories']}' if model_name == 'CategoricalVAE' and 'num_categories' in param.keys() else ''

            run_dir = 'Experiments/Synth_Portf/VAE/'+model_name+'_Enc'+encoder_name+'Latent'+str(param['latent_dim'])+cat_suffix+'_Dec'+decoder_name

            model_config = {
                'input_dim': 51,
                'enc_hidden_dims': param['enc_hidden_dims'],
                'latent_dim': param['latent_dim'],
                'dec_hidden_dims': param['dec_hidden_dims'],
                'output_split_sections': dataset.client_feat_split_sections,
                'split_section_types': dataset.client_feat_types,
                'loss_map': {'numerical': 'mse', 'categorical': 'cross_entropy', 'distribution': 'symmetric_kl', 'score': 'symmetric_kl', 'multiclass': 'binary_cross_entropy'},  #{'all': 'mse'}
                'beta_kl': 1
            }
            if model_name == 'CategoricalVAE' and 'num_categories' in param.keys():
                model_config['num_categories'] = param['num_categories']

            num_epochs = 50
            initial_beta_kl = 0.01
            beta_kl_max = 1
            beta_kl_growth_rate = math.log(beta_kl_max/initial_beta_kl) / (num_epochs*0.5)  # reverse anneal to reach beta_kl at around 50% of training (usually between 30%-50%)

            model = LitVAE(
                model_name, 
                model_config, 
                learning_rate = 1e-3,
                initial_temperature = 1.0,
                min_temperature = 0.1,
                temperature_decay_rate = 0.05,
                initial_beta_kl = initial_beta_kl,
                beta_kl_max = beta_kl_max ,
                beta_kl_growth_rate = beta_kl_growth_rate)
            trainer = L.Trainer(accelerator="gpu", max_epochs=num_epochs, check_val_every_n_epoch=1, default_root_dir=run_dir)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
            log_dir = trainer.logger.log_dir

            eval_model(
                base_model = model, 
                train_dataloader = train_loader,
                val_dataloader = val_loader,
                log_dir = log_dir)

    else:
        ckpt_path = 'Experiments/Synth_Portf/VAE/GaussianVAE_EncoderMLP32-32_Latent32_DecoderLin_MixedLoss/lightning_logs/version_0/checkpoints/epoch=49-step=1800.ckpt'
        #Load model
        model = LitVAE.load_from_checkpoint(ckpt_path)
        log_dir = Path(ckpt_path)
        log_dir = log_dir.parent.parent.absolute()

        eval_model(
            base_model = model, 
            train_dataloader = train_loader,
            val_dataloader = val_loader,
            log_dir = log_dir)
        
        # log_dir = 'Experiments/Synth_Portf/VAE/Baseline_NoTransform'
        # model_config = {
        #     'input_dim': 51,
        #     'enc_hidden_dims': [],
        #     'latent_dim': 51,
        #     'dec_hidden_dims': [],
        #     'output_split_sections': [51],
        #     'split_section_types': ['numerical'],
        #     'loss_map': {'numerical': 'mse'}
        # }

        # model = LitVAE(
        #     'IdentityVAE', 
        #     model_config)
        # eval_model(
        #     base_model = model, 
        #     train_dataloader = train_loader,
        #     val_dataloader = val_loader,
        #     log_dir = log_dir)

    
    
    # # Client embeddings
    # train_client_emb = []
    # train_targets = []
    # val_client_emb = []
    # val_targets = []
    # with torch.no_grad():
    #     model = model.eval().cuda()
    #     for batch in train_loader:
    #         x = model.vae.encode_reparametrize(batch[2].cuda()).cpu()
    #         if isinstance(model.vae, CategoricalVAE):
    #             x = x.flatten(1)  # flatten logits of categorical dimension
    #         elif isinstance(model.vae, GaussianVAE):
    #             x = x[0]  # use means
    #         elif isinstance(model.vae, PoissonVAE):
    #             x = x  # use logdr
    #         train_client_emb.append(batch[5].to('cpu'))  #client latents
    #         train_client_emb.append(x.to('cpu'))
    #     for batch in val_loader:
    #         x = model.vae.encode_reparametrize(batch[2].cuda()).cpu()
    #         if isinstance(model.vae, CategoricalVAE):
    #             x = x.flatten(1)  # flatten logits of categorical dimension
    #         elif isinstance(model.vae, GaussianVAE):
    #             x = x[0]  # use means
    #         elif isinstance(model.vae, PoissonVAE):
    #             x = x  # use logdr
    #         val_targets.append(batch[5].to('cpu'))  #client latents
    #         val_client_emb.append(x.to('cpu'))
    # train_client_emb = torch.cat(train_client_emb, dim=0)
    # train_targets = torch.cat(train_targets, dim=0)
    # val_client_emb = torch.cat(val_client_emb, dim=0)
    # val_targets = torch.cat(val_targets, dim=0)
    # target_labels = list(dataset.client_latents_df.iloc[:,2:].columns)
    # plot_embeddings(run_dir, train_client_emb, train_targets, target_labels, dataset.client_latents_map)
    # # if isinstance(model, CategoricalVAE) or isinstance(model, PoissonVAE):
    # #     plot_activation_histogram()

    # # Train Linear Classifier on client latents






