import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pystan

from bayestestimation.stan.model import _stan_model

class BayesTEstimation:

    def __init__(self, alpha=0, beta=1, delta=0.001, gamma=0.001, epsilon=30):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self._compile_model()

    def _compile_model(self):
        self.model = pystan.StanModel(model_code=_stan_model())

    def _create_data(self, x, y):
        return {
                'X': x,
                'Y': y,
                'n_x': len(x),
                'n_y': len(y),
                'alpha': self.alpha,
                'beta': self.beta,
                'delta': self.delta,
                'gamma': self.gamma,
                'epsilon': self.epsilon
                }

    def fit_model(self, x, y):
        self.fit = self.model.sampling(data=self._create_data(x, y))
        
    def _estimate_delta(self):
        return self.fit.extract('mu_y')['mu_y'] - self.fit.extract('mu_x')['mu_x']

    def _calculate_quantiles(self, d, mean, quantiles):
        # Calculate mean and quantiles
        q = np.quantile(d, quantiles)
        if mean is True:
            q = np.append(q, np.mean(d))    
        return q

    def get_posteriors(self):
        posteriors = self.fit.extract()
        return {
                'mu_x': posteriors['mu_x'],
                'mu_y': posteriors['mu_y'],
                'delta': self._estimate_delta(),
                'sigma_x': posteriors['mu_x'],
                'sigma_y': posteriors['mu_x'],
                'nu': posteriors['mu_x']
                }

    def quantile_summary(self):
        return print('hello')

    def kde_plot(self):

        quantiles = [0.025, 0.975]
        parameter_list = [
                         'mu_x',
                         'mu_y',
                         'delta',
                         'sigma_x',
                         'sigma_y',
                         'nu'
                         ]

        fig, axes = plt.subplots(2, 3, figsize=(15,10))

        k = 0
        for i in range(0, 2):
            for j in range(0, 3):
                k_str = parameter_list[k]
                if k_str == 'delta':
                    a = self._estimate_delta()
                else:
                    a = self.fit.extract(k_str)[k_str]

                sns.kdeplot(a, ax = axes[i, j])
                axes[i, j].set(xlabel=k_str, ylabel='density')
                x, y = axes[i, j].lines[0].get_data()
                q = self._calculate_quantiles(a, mean=False, quantiles=quantiles)
                axes[i, j].fill_between(x, y, where=((x >= q[0]) & (x <= q[1])), alpha=0.2)
                axes[i, j].fill_between(x, y, where=((x <= q[0]) | (x >= q[1])), alpha=0.1)
                if k_str == 'delta':
                    axes[i, j].axvline(0, ls='--', color='red')
                k = k + 1


    
