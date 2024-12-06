import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import gamma, norm, truncnorm
from scipy.optimize import minimize

class PortfolioSimulator:
    def __init__(self):
        self.SECTORS = ['Technology', 'Healthcare', 'Finance', 'Consumer', 'Energy', 
                       'Industrial', 'Materials', 'Utilities', 'Real Estate']
        self.COUNTRIES = ['US', 'UK', 'Germany', 'Japan', 'China', 'Canada', 
                         'Australia', 'France', 'Switzerland', 'Netherlands']
        self.ASSET_TYPES = ['Stock', 'Bond', 'ETF', 'REIT']
        self.MARKET_CAPS = ['Small', 'Mid', 'Large']
        self.INVESTMENT_STYLES = ['Dividend', 'Balanced', 'Growth']
        self.AGE_GROUPS = ['25-35', '35-45', '45-55', '55-65', '65+']
        self.INCOME_LEVELS = ['Medium', 'High', 'Very High']
        self.GENDERS = ['Male', 'Female', 'Other']
        self.DIVERSITY_LEVEL = ['Divers', 'Focussed']
        
        # Define country-specific characteristics
        self.COUNTRY_PROFILES = {
            'US': {
                'tech_affinity': 0.8,
                'international_exposure': 0.7,
                'risk_tolerance': 0.7,
                'esg_preference': 0.6
            },
            'Germany': {
                'tech_affinity': 0.6,
                'international_exposure': 0.8,
                'risk_tolerance': 0.6,
                'esg_preference': 0.8
            },
            'Japan': {
                'tech_affinity': 0.7,
                'international_exposure': 0.4,
                'risk_tolerance': 0.4,
                'esg_preference': 0.7
            }
        }
        
        # Default values for other countries
        self.DEFAULT_COUNTRY_PROFILE = {
            'tech_affinity': 0.5,
            'international_exposure': 0.5,
            'risk_tolerance': 0,
            'esg_preference': 0.5
        }
        
        # New constants for portfolio values
        self.PORTFOLIO_VALUE_PARAMS = {
            'Medium': {'mean': 100000, 'std': 50000, 'min': 10000, 'max': 500000},
            'High': {'mean': 500000, 'std': 200000, 'min': 100000, 'max': 2000000},
            'Very High': {'mean': 2000000, 'std': 1000000, 'min': 500000, 'max': 10000000}
        }
        
        # Asset return and risk parameters
        self.ASSET_RETURN_PARAMS = {
            'Stock': {'mean': 0.10, 'std': 0.20},
            'Bond': {'mean': 0.04, 'std': 0.05},
            'ETF': {'mean': 0.08, 'std': 0.15},
            'REIT': {'mean': 0.07, 'std': 0.18}
        }
        
        # Sector correlation base parameters
        self.SECTOR_CORRELATIONS = {
            'Technology': {'Healthcare': 0.3, 'Finance': 0.5, 'Consumer': 0.4},
            'Healthcare': {'Finance': 0.3, 'Consumer': 0.2},
            'Finance': {'Consumer': 0.6},
            'Energy': {'Industrial':0.5}
            # Add more sector correlations...
        }


    def generate_correlation_matrix(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        """Generate a realistic correlation matrix for assets. Used for portfolio optimization"""
        n_assets = len(assets_df)
        base_corr = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                asset1, asset2 = assets_df.iloc[i], assets_df.iloc[j]
                
                # Base correlation from sector relationship
                sector_corr = self.SECTOR_CORRELATIONS.get(asset1['sector'], {}).get(
                    asset2['sector'], 0.3)  # Default correlation if not specified
                
                # Modify correlation based on asset types
                type_modifier = 1.0
                if asset1['asset_type'] != asset2['asset_type']:
                    type_modifier = 0.7  # Lower correlation between different asset types
                
                # Modify correlation based on geography
                geo_modifier = 1.0
                if asset1['country'] != asset2['country']:
                    geo_modifier = 0.8  # Lower correlation between different countries
                
                # Calculate final correlation with some random noise
                correlation = sector_corr * type_modifier * geo_modifier
                correlation += np.random.normal(0, 0.05)  # Add small random variation
                correlation = np.clip(correlation, -1, 1)
                
                base_corr[i,j] = correlation
                base_corr[j,i] = correlation
        
        # Ensure the matrix is positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(base_corr)
        eigenvals[eigenvals < 0] = 0
        base_corr = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return pd.DataFrame(base_corr, 
                          index=assets_df['asset_id'],
                          columns=assets_df['asset_id'])
    
    # Generate asset correlation matrix
    def generate_preference_asset_correlation_matrix(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        num_assets = len(assets_df)
        # Simulate a random correlation matrix
        random_matrix = np.random.rand(num_assets, num_assets)
        # Make it symmetric
        corr_matrix = (random_matrix + random_matrix.T) / 2
        # Set diagonals to 1
        np.fill_diagonal(corr_matrix, 1)
        # Ensure positive semi-definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues[eigenvalues < 0] = 0
        corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return pd.DataFrame(corr_matrix, index=assets_df['asset_id'], columns=assets_df['asset_id'])

    def get_portfolio_size(self, client_attributes: Dict) -> int:
        """
        Determine portfolio size based on client attributes using a gamma distribution.
        Different client segments have different average portfolio sizes.
        """
        # Base parameters for gamma distribution
        base_shape = 2.5  # Controls the shape of the distribution
        
        # Adjust scale based on client attributes
        base_scale = 5.0  # This gives a reasonable starting mean
        
        # Modify scale based on income level
        income_multiplier = {
            'Medium': 1.0,
            'High': 1.5,
            'Very High': 2.0
        }[client_attributes['income_level']]
        
        # Age group modifier - older clients tend to have more diverse portfolios
        age_multiplier = {
            '25-35': 0.8,
            '35-45': 1.0,
            '45-55': 1.2,
            '55-65': 1.3,
            '65+': 1.4
        }[client_attributes['age_group']]
        
        # Risk tolerance modifier - aggressive investors tend to have more positions
        risk_multiplier = 1 + (client_attributes['risk_tolerance'] - 0.5)/2
        
        # Calculate final scale
        final_scale = base_scale * income_multiplier * age_multiplier * risk_multiplier
        
        # Generate portfolio size from gamma distribution
        size = int(gamma.rvs(base_shape, scale=final_scale))
        
        # Ensure reasonable bounds
        return max(min(size, 100), 3)  # Between 3 and 100 positions

    def get_client_preferences(self, client_attributes: Dict) -> Dict:
        """
        Generate client preferences based on their attributes.
        Returns a dictionary of preference weights for different investment factors.
        """
        country_profile = self.COUNTRY_PROFILES.get(
            client_attributes['country'], 
            self.DEFAULT_COUNTRY_PROFILE
        )
        
        # Base preferences
        preferences = {
            'tech_affinity': country_profile['tech_affinity'],
            'international_exposure': country_profile['international_exposure'],
            'risk_tolerance': self.get_truncated_normal({'mean': country_profile['risk_tolerance'], 'std':0.2, 'min':0, 'max':1}),
            'esg_preference': country_profile['esg_preference'],
            'dividend_preference': 0.5,
            'growth_preference': 0.5,
            'small_cap_affinity': 0.5,
            'home_bias': 0.5
        }
        
        # Modify based on age
        age_modifier = {
            '25-35': {'tech_affinity': 0.2, 'risk_tolerance': 0.2, 'growth_preference': 0.2},
            '35-45': {'tech_affinity': 0.1, 'risk_tolerance': 0.1, 'growth_preference': 0.1},
            '45-55': {'tech_affinity': 0, 'risk_tolerance': 0, 'growth_preference': 0},
            '55-65': {'tech_affinity': -0.1, 'dividend_preference': 0.1, 'risk_tolerance': -0.1},
            '65+': {'tech_affinity': -0.2, 'dividend_preference': 0.2, 'risk_tolerance': -0.2}
        }[client_attributes['age_group']]
        
        # Apply age modifications
        for key, modifier in age_modifier.items():
            preferences[key] = np.clip(preferences[key] + modifier, 0, 1)
        
        # Gender-based modifications (based on statistical trends, not stereotypes)
        if client_attributes['gender'] == 'Female':
            preferences['risk_tolerance'] *= 0.9  # Slightly more risk-averse on average
            preferences['esg_preference'] *= 1.1  # Slightly higher ESG preference on average
        
        # Income level modifications
        income_modifier = {
            'Medium': {'international_exposure': -0.1, 'small_cap_affinity': 0.1},
            'High': {'international_exposure': 0, 'small_cap_affinity': 0},
            'Very High': {'international_exposure': 0.1, 'small_cap_affinity': -0.1}
        }[client_attributes['income_level']]
        
        # Apply income modifications
        for key, modifier in income_modifier.items():
            preferences[key] = np.clip(preferences[key] + modifier, 0, 1)
        
        # Add random noise to each preference
        for key in preferences:
            noise = np.random.normal(0, 0.05)  # Small random variation
            preferences[key] = np.clip(preferences[key] + noise, 0, 1)
        
        return preferences

    def generate_clients(self, num_clients: int) -> pd.DataFrame:
        """Generate synthetic client data with enhanced attributes."""
        clients = []
        
        for _ in range(num_clients):
            age_group = np.random.choice(self.AGE_GROUPS)
            country = np.random.choice(self.COUNTRIES)
            
            # Base client attributes
            client = {
                'client_id': f'C{_:07d}',
                'age_group': age_group,
                'gender': np.random.choice(self.GENDERS),
                'country': country,
                'income_level': np.random.choice(self.INCOME_LEVELS),
                'investment_style': np.random.choice(self.INVESTMENT_STYLES),
                'diversity_level': np.random.choice(self.DIVERSITY_LEVEL, p=[0.9,0.1])
            }
            
            # Generate preferences based on attributes
            preferences = self.get_client_preferences(client)
            
            # Add preferences to client data
            client.update(preferences)
            
            # Generate preferred sectors based on preferences
            num_preferred_sectors = np.random.randint(1, 4)
            sector_weights = []
            for sector in self.SECTORS:
                weight = preferences['tech_affinity'] if sector == 'Technology' else 0.5
                sector_weights.append(weight)
            sector_weights = np.array(sector_weights) / sum(sector_weights)
            
            client['preferred_sectors'] = np.random.choice(
                self.SECTORS, 
                size=num_preferred_sectors, 
                p=sector_weights,
                replace=False
            ).tolist()
            
            clients.append(client)
        
        return pd.DataFrame(clients)

    def generate_assets(self, num_assets: int) -> pd.DataFrame:
        """Generate synthetic asset data."""
        assets = []
        
        for _ in range(num_assets):
            asset_type = np.random.choice(self.ASSET_TYPES)
            
            # Adjust dividend yield based on asset type
            if asset_type == 'Bond':
                dividend_yield = np.random.uniform(0.02, 0.05)
            elif asset_type == 'REIT':
                dividend_yield = np.random.uniform(0.03, 0.07)
            else:
                dividend_yield = np.random.uniform(0, 0.04)
            
            asset = {
                'asset_id': f'A{_:05d}',
                'asset_type': asset_type,
                'sector': np.random.choice(self.SECTORS),
                'country': np.random.choice(self.COUNTRIES),
                'market_cap': np.random.choice(self.MARKET_CAPS),
                'dividend_yield': dividend_yield,
                'volatility_risk': np.random.uniform(0, 1),
                'esg_score': np.random.uniform(50, 100)
            }
            assets.append(asset)
        
        return pd.DataFrame(assets)

    def calculate_preference_score(self, client: pd.Series, asset: pd.Series) -> float:
        """Calculate preference score between a client and an asset using enhanced preferences."""
        score = 0.5  # Base score
        
        # Home bias based on client's international exposure preference
        if client['country'] == asset['country']:
            score += 0.1 * (1 - client['international_exposure'])
        else:
            score += 0.1 * client['international_exposure']
        
        # Sector preference with technology affinity
        if asset['sector'] in client['preferred_sectors']:
            sector_bonus = 0.15
            if asset['sector'] == 'Technology':
                sector_bonus *= (1 + client['tech_affinity'])
            score += sector_bonus
        
        # Risk alignment 
        volatility_penalty = abs(client['risk_tolerance'] - asset['volatility_risk'])
        score -= volatility_penalty
        
        # Investment style alignment with preferences
        if client['investment_style'] == 'Dividend':
            score += 0.1 * client['dividend_preference'] * asset['dividend_yield'] / 0.05
        elif client['investment_style'] == 'Growth':
            score += 0.1 * client['growth_preference'] * (1 - asset['dividend_yield'] / 0.05)
        
        # Market cap preference
        if asset['market_cap'] == 'Small':
            score += 0.1 * client['small_cap_affinity']
        
        # ESG preference
        score += 0.1 * client['esg_preference'] * (asset['esg_score'] / 100)
        
        return score

    def get_truncated_normal(self, params: Dict) -> float:
        """Generate a truncated normal distribution value."""
        mean, std = params['mean'], params['std']
        min_val, max_val = params['min'], params['max']
        
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std)

    def calculate_portfolio_metrics(self, weights: np.array, returns: np.array, 
                                 cov_matrix: np.array) -> Tuple[float, float]:
        """Calculate portfolio return and risk."""
        portfolio_return = np.sum(returns * weights)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return portfolio_return, portfolio_risk

    def optimize_portfolio(self, returns: np.array, cov_matrix: np.array, 
                         risk_tolerance: float) -> np.array:
        """Optimize portfolio weights using mean-variance optimization."""
        n_assets = len(returns)
        
        def objective(weights):
            port_return, port_risk = self.calculate_portfolio_metrics(
                weights, returns, cov_matrix)
            # Utility function combining return and risk based on risk tolerance
            return -(port_return - (1-risk_tolerance) * port_risk)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Asset allocation bounds
        bounds = tuple((0, 0.4) for _ in range(n_assets))  # Max 40% in single asset
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                        bounds=bounds, constraints=constraints)
        
        return result.x

    def generate_portfolio_data(self, num_clients: int, 
                              num_assets: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate complete dataset with portfolio values and allocations."""
        
        # Generate base data
        clients_df = self.generate_clients(num_clients)
        assets_df = self.generate_assets(num_assets)
        
        # Generate correlation matrix and expected returns
        correlation_matrix = self.generate_correlation_matrix(assets_df)
        preference_correlation_matrix = self.generate_preference_asset_correlation_matrix(assets_df)
        
        portfolios = []
        counter = 0
        for _, client in clients_df.iterrows():
            # Generate portfolio value based on income level
            portfolio_value = self.get_truncated_normal(
                self.PORTFOLIO_VALUE_PARAMS[client['income_level']])
            
            # Determine portfolio size
            portfolio_size = self.get_portfolio_size(client)

            # Calculate preference scores for all assets
            scores = assets_df.apply(lambda x: self.calculate_preference_score(
                client, x), axis=1)
            # Convert scores to selection probabilities
            check_diversity = client['diversity_level'] == 'Divers'
            alpha = 1.0 if check_diversity else 3  # Scaling parameter
            selection_probs = np.exp(np.array(scores) * alpha)
            selection_probs /= selection_probs.sum()
            
            # Initialize selected assets list
            selected_asset_ids = []
            asset_ids = assets_df['asset_id'].values
            
            # Sample Assets
            while len(selected_asset_ids) < min(portfolio_size, num_assets):
                # Sample an asset
                candidate_asset_id = np.random.choice(
                    asset_ids, p=selection_probs)
                # Check diversification
                if selected_asset_ids and check_diversity:
                    # Compute average correlation with selected assets
                    correlations = correlation_matrix.loc[selected_asset_ids, candidate_asset_id]
                    avg_correlation = correlations.mean()
                    if avg_correlation < 0.7:  # Threshold for diversification
                        selected_asset_ids.append(candidate_asset_id)
                else:
                    selected_asset_ids.append(candidate_asset_id)
                # Remove selected asset from future consideration
                index = np.where(asset_ids == candidate_asset_id)[0][0]
                asset_ids = np.delete(asset_ids, index)
                selection_probs = np.delete(selection_probs, index)
                # Adjust selection probabilities based on preference asset correlations
                pref_correlation = preference_correlation_matrix.loc[candidate_asset_id, asset_ids]
                selection_probs *= pref_correlation
                selection_prob_sum = selection_probs.sum()

                if selection_prob_sum > 0:
                    selection_probs /= selection_prob_sum
                else:
                    break  # No more assets to select
                
                
            
            # Assign allocation percentages (weights)
            weights = np.random.dirichlet(np.ones(len(selected_asset_ids)), size=1)[0]
            # Ensure weights sum to 1
            weights /= weights.sum()
            
            # Calculate allocation amounts
            allocations = weights * portfolio_value
            
            # Add to portfolio_data
            for asset_id, weight, allocation in zip(selected_asset_ids, weights, allocations):
                portfolio = {
                    'client_id': client['client_id'],
                    'asset_id': asset_id,
                    'allocation_percentage': weight,
                    'allocation_amount': allocation
                }
                portfolios.append(portfolio)

            counter += 1
            if counter%100 == 0:
                print(f'Portfolios generated: {counter}')
        
        portfolios_df = pd.DataFrame(portfolios)

        # Add preference correlation matrix to assets_df as additional info
        assets_df.attrs['correlation_matrix'] = preference_correlation_matrix
        
        return clients_df, assets_df, portfolios_df


if __name__ == '__main__':

    sim_run = 'Default_C10000_A500'
    if not os.path.exists('Datasets/Synth_Portf/'+sim_run):
        os.makedirs('Datasets/Synth_Portf/'+sim_run)
    simulator = PortfolioSimulator()
    print('Start Generating Portfolios...')
    clients_df, assets_df, portfolios_df = simulator.generate_portfolio_data(
        num_clients=10000, 
        num_assets=500
    )

    # Access correlation matrix
    correlation_matrix = assets_df.attrs['correlation_matrix']

    # Save to separate files
    clients_df.to_csv('Datasets/Synth_Portf/'+sim_run+'/clients.csv')
    assets_df.to_csv('Datasets/Synth_Portf/'+sim_run+'/assets.csv')
    portfolios_df.to_csv('Datasets/Synth_Portf/'+sim_run+'/portfolios.csv')
    correlation_matrix.to_csv('Datasets/Synth_Portf/'+sim_run+'/correlations.csv')
