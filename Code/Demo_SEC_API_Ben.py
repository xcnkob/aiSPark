# -*- coding: utf-8 -*-
"""

"""
import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Recommender_Models import EmbeddingRecommender,MLPRecommender,LitRecommenderWrapper
import lightning as L


class SecAPIDataset(Dataset):
    def __init__(self, asset_freq_threshold, portfolio_asset_threshold, force_reload=False, edge_mask_ratio=0.2, num_asset_freq_buckets=10):
        json_folder_path = 'C:/Users/ben.koch/Desktop/aSpark Recommender Systems/Datasets/SEC_API_Test Daten/'
        pre_processed_filename = 'dataframe.pkl'
        if os.path.exists(os.path.join(json_folder_path, pre_processed_filename)) and not force_reload:
            print('Loading processed JSON Files')
            portfolios = pd.read_pickle(os.path.join(json_folder_path, pre_processed_filename))
        else:
            print('Start processing JSON Files')
            FUND_ID = 0
            portfolios = []
            relv_columns = ['FUND_ID', 'identifiers.isin.value', 'assetCat', 'invCountry', 'payoffProfile']
            file_counter=0
            for filename in os.listdir(json_folder_path):
                if filename.startswith("SEC_Data_") and filename.endswith(".json"):
                    json_file_path = os.path.join(json_folder_path, filename)
                    # JSON-Datei öffnen und Daten laden
                    with open(json_file_path, 'r') as json_file:
                        response = json.load(json_file)
                    for filing in response["filings"]:
                        filing["FUND_ID"] = FUND_ID
                        FUND_ID += 1
                    df = pd.json_normalize(response.get("filings", []),record_path='invstOrSecs',meta=['FUND_ID',['filerInfo','seriesClassInfo','seriesId'],['genInfo','regFileNumber']], max_level=4,errors='ignore')
                    # Populate ISIN for missing values
                    df['identifiers.isin.value'] = df.get('identifiers.isin.value', df.get('identifiers.other.value', df.get('identifiers.other.value')))
                    df['identifiers.isin.value'] = df['identifiers.isin.value'].replace('N/A',pd.NA).replace('n/a',pd.NA).fillna(df.get('identifiers.ticker.value',pd.NA)).replace('N/A',pd.NA).replace('n/a',pd.NA).fillna(df.get('identifiers.other.value', pd.NA)).replace('N/A',pd.NA).replace('n/a',pd.NA)
                    # Drop rows with empty ISIN
                    df = df[relv_columns].dropna(subset='identifiers.isin.value').drop_duplicates(subset='identifiers.isin.value')
                    portfolios.append(df)
                    file_counter += 1
                    if file_counter%50==0:
                        print(f'Files processed: {file_counter}')
            portfolios = pd.concat(portfolios, ignore_index=True)
            portfolios.to_pickle(os.path.join(json_folder_path, pre_processed_filename))
        self.num_asset_freq_buckets = num_asset_freq_buckets
        self.edge_mask_ratio = edge_mask_ratio
        print('Start preprocessing Dataset')
        self.portfolios = self._preprocess(portfolios, asset_freq_threshold, portfolio_asset_threshold)
        print(f'Init Dataset End. Number of unique assets: {len(self.asset_key_map)}. Number of unique portfolios: {self.num_portfolios}')
        
    def __len__(self):
        return self.num_portfolios
    
    def __getitems__(self, indices):
        sel_idx = self.portfolios['FUND_IDX'].isin(indices)
        data = self.portfolios.loc[sel_idx]
        portfolio_indices = torch.tensor(pd.factorize(data['FUND_ID'])[0])
        asset_indices = torch.tensor(data['identifiers.isin.value'].values)
        add_feat = torch.tensor(data.iloc[:,3:].values)
        mask = self.val_mask[sel_idx.values]
        val_mask = mask==1
        train_mask = mask==0
        return portfolio_indices, asset_indices, add_feat, train_mask, val_mask
    
    def _preprocess(self, portfolios, asset_freq_threshold, portfolio_asset_threshold):
        while True:
            # Filter out assets that have fewer overall occurences than the threshold
            asset_counts = portfolios.groupby('identifiers.isin.value')['FUND_ID'].count()
            valid_assets = asset_counts[asset_counts >= asset_freq_threshold].index
            portfolios = portfolios[portfolios['identifiers.isin.value'].isin(valid_assets)]
            # Filter out portfolios that have fewer assets than the threshold
            df_prev_size = portfolios['FUND_ID'].size
            portfolio_counts = portfolios.groupby('FUND_ID')['identifiers.isin.value'].count()
            valid_portfolios = portfolio_counts[portfolio_counts >= portfolio_asset_threshold].index
            portfolios = portfolios[portfolios['FUND_ID'].isin(valid_portfolios)]
            if portfolios['FUND_ID'].size == df_prev_size:
                # When no more portfolios are deleted, we're done. Otherwise check if now we need to delete again some assets that fall below threshold
                break
        # Get indices for all values instead of strings and store maps
        asset_indices, asset_keys = pd.factorize(portfolios['identifiers.isin.value'])
        portfolios['identifiers.isin.value'] = asset_indices
        self.asset_key_map = dict(zip(asset_keys, asset_indices))
        self.asset_counts = portfolios.groupby('identifiers.isin.value')['FUND_ID'].count()

        # Get new consecutive Index for FUND
        portfolio_indices= pd.factorize(portfolios['FUND_ID'])[0]
        portfolios.insert(0, 'FUND_IDX', portfolio_indices, True)
        self.num_portfolios = portfolios['FUND_ID'].nunique()

        asset_cat_indices, asset_cat_keys = pd.factorize(portfolios['assetCat'])
        portfolios['assetCat'] = asset_cat_indices
        self.asset_cat_key_map = dict(zip(asset_cat_keys, asset_cat_indices))

        inv_country_indices, inv_country_keys = pd.factorize(portfolios['invCountry'])
        portfolios['invCountry'] = inv_country_indices
        self.inv_country_key_map = dict(zip(inv_country_keys, inv_country_indices))

        payoff_profile_indices, payoff_profile_keys = pd.factorize(portfolios['payoffProfile'])
        portfolios['payoffProfile'] = payoff_profile_indices
        self.payoff_profile_key_map = dict(zip(payoff_profile_keys, payoff_profile_indices))

        # Calculate asset frequencies (in terms of the percentage of portfolios they occur in)
        asset_counts = torch.tensor(list(self.asset_counts.items()))[:,1]
        self.asset_frequencies = asset_counts / self.num_portfolios

        # Create asset frequency buckets
        bucket_counts, bucket_upper_edges = torch.histogram(self.asset_frequencies, self.num_asset_freq_buckets)
        self.bucket_upper_edges = bucket_upper_edges[1:]
        self.bucket_assignments = torch.bucketize(self.asset_frequencies, self.bucket_upper_edges)

        # Create stratified mask for validation
        self.val_mask = torch.zeros(portfolios['FUND_ID'].count())
        asset_indices = torch.tensor(portfolios['identifiers.isin.value'].values)
        for i, bucket_size in enumerate(bucket_counts):
            if bucket_size == 0:
                continue
            # select edge_mask_ratio of edges of assets in the current bucket and remove them (=add to val_mask)
            relv_assets_indices = torch.where(self.bucket_assignments==i)[0]
            relv_edge_indices = torch.where(torch.isin(asset_indices, relv_assets_indices))[0]
            num_masked = int(relv_edge_indices.shape[0] * self.edge_mask_ratio)
            masked_edge_indices = np.random.choice(relv_edge_indices, num_masked, replace=False)
            self.val_mask[masked_edge_indices] = 1

        return portfolios
    
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

def plot_dataset(portfolio_asset_threshold=5, asset_freq_threshold=5):
    import matplotlib.pyplot as plt
    dataset = SecAPIDataset(asset_freq_threshold=asset_freq_threshold, portfolio_asset_threshold=portfolio_asset_threshold)
        
    asset_group = dataset.portfolios.groupby('identifiers.isin.value')['FUND_ID'].count()
    portfolio_group = dataset.portfolios.groupby('FUND_ID')['identifiers.isin.value'].count()
    x = []
    asset_counts = []
    portfolio_counts = []
    for i in range(min(portfolio_asset_threshold, asset_freq_threshold), 200):
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

def plot_embedding_norms(save_dir, embedding_model, dataset):
    assert isinstance(embedding_model, EmbeddingRecommender), 'embedding model must be a model of class EmbeddingRecommender'
    import matplotlib.pyplot as plt
    feat_embedding = embedding_model.feat_embedding.weight.detach().cpu()
    target_embedding = embedding_model.target_embedding.weight.detach().cpu()
    feat_embedding_norm, feat_sort_indices = feat_embedding.norm(dim=1).sort(descending=True)
    target_embedding_norm, target_sort_indices = target_embedding.norm(dim=1).sort(descending=True)
    # get asset occurences as well
    asset_occurences = dataset.portfolios.groupby('identifiers.isin.value')['FUND_ID'].count()
    feat_asset_occurences = asset_occurences.iloc[feat_sort_indices].to_list()
    target_asset_occurences = asset_occurences.iloc[target_sort_indices].to_list()

    f, ax = plt.subplots(2)
    color = 'tab:red'
    ax[0].plot(feat_embedding_norm, color=color)
    ax[0].title.set_text('Feature Embeddding Norms')
    ax[0].set_xlabel('Norm-sorted asset index')
    ax[0].set_ylabel('Norm', color=color)
    ax[0].tick_params(axis='y', labelcolor=color)
    ax2 = ax[0].twinx()
    color = 'tab:blue'
    ln2 = ax2.plot(feat_asset_occurences, color=color)
    ax2.set_ylabel('Asset occurences', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:red'
    ax[1].plot(target_embedding_norm, color=color)
    ax[1].title.set_text('Target Embeddding Norms')
    ax[1].set_xlabel('Norm-sorted asset index')
    ax[1].set_ylabel('Norm', color=color)
    ax2 = ax[1].twinx()
    color = 'tab:blue'
    ln2 = ax2.plot(target_asset_occurences, color=color)
    ax2.set_ylabel('Asset occurences', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    f.tight_layout()
    f.savefig(save_dir+'/embedding_norms.png')
    # plt.show()

    # self_scores = torch.matmul(feat_embedding.unsqueeze(1),target_embedding.unsqueeze(-1)).squeeze(1).squeeze(1)
    # self_scores, _ = self_scores.sort()
    # f, ax = plt.subplots(1)
    # ax.plot(self_scores)
    # plt.show()

def plot_tsne_embeddings(save_dir, feat_embeddings, target_embeddings, title):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    print('Start T-SNE embeddings')
    #todo: use dot product as metric, + use portfolio embeddings instead of (only) feature embeddings
    feat_c = ['blue']*feat_embeddings.shape[0]
    if target_embeddings is not None:
        target_c = ['red']*target_embeddings.shape[0]
        c = feat_c + target_c
        emb = torch.cat([feat_embeddings, target_embeddings])
    else:
        emb = feat_embeddings
        c = feat_c
    emb /= emb.norm(2, dim=1, keepdim=True)  #L2-normalisation

    emb = TSNE(n_components=2, learning_rate='auto', perplexity=30, metric='cosine').fit_transform(emb.numpy())
    f, ax = plt.subplots(1)
    ax.scatter(emb[:,0],emb[:,1], c=c, s=1)
    ax.title.set_text(title)
    f.tight_layout()
    f.savefig(save_dir+'/' + title + ' t-sne.png')

def plot_mlp_portf_embeddings(save_dir, mlp, dataloader):
    import matplotlib.pyplot as plt
    emb_x = []
    emb_y = []
    with torch.no_grad():
        mlp = mlp.to('cpu')
        for batch in dataloader:
            embeddings = mlp.forward_to_final_layer(batch[1], batch[0])
            assert embeddings.shape[1] == 2, 'Dimension of final layer must be 2'
            embeddings = embeddings.numpy()
            emb_x.extend(embeddings[:,0])
            emb_y.extend(embeddings[:,1])
        target_emb = mlp.pred_layer[0].weight.detach().numpy()
        if hasattr(mlp.pred_layer[0], 'alpha'):
            title_append = f' (alpha={round(mlp.pred_layer[0].alpha.item(), 2)}, beta={round(mlp.pred_layer[0].beta.item(),2)})'
        else:
            title_append = ''
    f, ax = plt.subplots(2)
    ax[0].scatter(emb_x,emb_y, alpha=0.5, marker='+')
    ax[0].title.set_text(f'Portfolio Embedddings MLP)')
    ax[1].scatter(target_emb[:,0], target_emb[:,1], alpha=0.5, marker='+')
    ax[1].title.set_text('Target Asset Embedddings MLP' + title_append)
    f.tight_layout()
    f.savefig(save_dir+'/Embedding_bottleneck_MLP.png')


if __name__ == '__main__':
    train_model = True
    resume_training = False
    visualize_dataset = False
  
    run_dir = 'Experiments/Embedding_Model/16_softmax_portfmean_w_priors'
    if not train_model or resume_training:
        ckpt_path = run_dir+'/lightning_logs/version_0/checkpoints/epoch=199-step=2200.ckpt'
    else:
        ckpt_path = None

    if visualize_dataset:
        plot_dataset()
        
    portfolio_asset_threshold = 5
    asset_freq_threshold = 25
    dataset = SecAPIDataset(asset_freq_threshold=asset_freq_threshold, portfolio_asset_threshold=portfolio_asset_threshold, num_asset_freq_buckets=10, edge_mask_ratio=0.2)
    train_loader, val_loader, _ = create_data_loaders(dataset=dataset, batch_size=256, val_split_ratio=0.1, test_split_ratio=0., num_workers=0)

    asset_counts = torch.tensor(dataset.asset_counts)
    if train_model:
        model_name = 'EmbeddingRecommender'
        if model_name == 'EmbeddingRecommender':
            model_config = {
                'num_assets': len(dataset.asset_key_map),
                'emb_dim': 16,
                'asset_key_index_map': dataset.asset_key_map,
                'probs_norm_method': 'softmax',
                'r_mask': 1,
                'asset_counts': asset_counts,
                'masking_strategy': 'random',
                'feat_embedding_max_norm': None,
                'target_embedding_max_norm': None,
                'scale_emb_grad_by_freq': False,
                'sparse_embedding': False,
                'portf_emb_aggr_method': 'Mean',
                'norm_asset_feat_embedding': False
            }
        elif model_name == 'MLPRecommender':
            model_config = {
                'num_assets': len(dataset.asset_key_map),
                'hidden_dim': 128,
                'num_hidden_layers': 2,
                'asset_key_index_map': dataset.asset_key_map,
                'probs_norm_method': 'softmax',
                'r_mask': 1,
                'asset_counts': asset_counts,
                'masking_strategy': 'random',
                'dropout_rate': 0., 
                'bottleneck_dim': 2,
                'norm_portf_emb_by_portf_size':True,
                'neg_target_ratio': 0.5
            }
        lit_model = LitRecommenderWrapper(model_name=model_name, model_config=model_config, ign_self_scores=True, eval_train_every_n_epoch=100)
        trainer = L.Trainer(accelerator="gpu", max_epochs=2000, check_val_every_n_epoch=100, default_root_dir=run_dir)
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=[val_loader, train_loader], ckpt_path=ckpt_path)
    else:
        #Load model
        lit_model = LitRecommenderWrapper.load_from_checkpoint(ckpt_path)
    if isinstance(lit_model.recommender, EmbeddingRecommender):
        plot_embedding_norms(run_dir, lit_model.recommender, dataset=dataset)
        plot_tsne_embeddings(run_dir, lit_model.recommender.feat_embedding.weight.detach().cpu(), lit_model.recommender.target_embedding.weight.detach().cpu(), title='Asset and Target Embeddings')
        # portfolio embeddings
        portf_emb = []
        with torch.no_grad():
            lit_model = lit_model.to('cpu')
            for batch in train_loader:
                portf_emb.append(lit_model.recommender.forward_feat_emb(batch[1], batch[0]))
        portf_emb = torch.cat(portf_emb, dim=0)
        plot_tsne_embeddings(run_dir, portf_emb, None, title='Portfolio Embeddings')
        plot_tsne_embeddings(run_dir, portf_emb, lit_model.recommender.target_embedding.weight.detach(), title='Portfolio and Target Embeddings')
    elif isinstance(lit_model.recommender, MLPRecommender):
        if lit_model.hparams.model_config['bottleneck_dim'] is not None:
            plot_mlp_portf_embeddings(run_dir, lit_model.recommender, dataloader=train_loader)



