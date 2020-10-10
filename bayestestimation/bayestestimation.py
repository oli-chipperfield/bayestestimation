import numpy as np
import pandas as pd
import arviz as az
import pystan as pystan

from bayestestimation.model.best import model

class BayesTEstimation:

    def __init__(self):
        self.stan_model = self._compile_model()
        
    def _compile_model(self):
        # Compiles the stan model in C++
        return pystan.StanModel(model_code=model, model_name='BEST')

    def _check_sample_posterior_inputs(self):
        # Checks that parameters are in the correct format
        types = ['list', 'ndarray', 'Series']
        mu_types = ['float', 'int']
        if ((type(self.x).__name__ not in types) or (type(self.y).__name__ not in types)):
            raise ValueError("type(x).__name__ and/or type(y).__name__ must be 'list', 'ndarray' or 'Series'")
        if self.n <= 0:
            raise ValueError("n must be a positive integer")
        if ((self.prior_alpha <= 0) or (self.prior_beta <= 0) or (self.prior_phi <= 0)):
            raise ValueError("the prior_alpha and/or prior_beta and/or prior_phi parameters must be > 0")
        if self.prior_mu is not None and (type(self.prior_mu).__name__ not in mu_types):
            raise ValueError("prior_mu must be None or type(prior_mu).__name__ must be 'float' or 'int'")       
        if self.prior_s is not None and self.prior_s <= 0:
            raise ValueError("prior_s must be None or must be > 0")
        if self.seed is not None and str(self.seed).isdigit() == False:
            raise ValueError("seed must be a positive integer or None")

    def _estimate_prior_mu(self):
        # Estimates a prior value for mu when one isn't provided
        return np.mean(np.concatenate([self.x, self.y]))

    def _estimate_prior_s(self):
        # Estimates a prior value for s when one isn't provided
        return np.std(np.concatenate([self.x, self.y]))

    def _build_input_dictionary(self):
        # Builds a dictionary input to apply to the model
        return {'x': self.x,
                'y': self.y,
                'n_x': self.n_x,
                'n_y': self.n_y,
                'mu': self.prior_mu,
                's': self.prior_s,
                'phi': self.prior_phi,
                'alpha': self.prior_alpha,
                'beta': self.prior_beta}

    def fit_posteriors(self, 
                       x, 
                       y, 
                       n=10000,
                       prior_alpha=0.001,
                       prior_beta=0.001,
                       prior_phi=(1/30),
                       prior_mu=None,
                       prior_s=None,
                       seed=None):
        '''
        a doc string
        '''
        self.x = x
        self.y = y
        self.n_x = len(x)
        self.n_y = len(y)
        self.n = n
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.prior_phi = prior_phi
        self.prior_mu = prior_mu
        self.prior_s = prior_s
        self.seed = seed
        self._check_sample_posterior_inputs()
        if self.prior_mu is None:
            self.prior_mu = self._estimate_prior_mu()
        if self.prior_s is None:
            self.prior_s = self._estimate_prior_s()
        self.fit = self.stan_model.sampling(data=self._build_input_dictionary(), 
                                            seed=self.seed,
                                            iter=int(np.ceil((self.n / 4) * 2)))


        




