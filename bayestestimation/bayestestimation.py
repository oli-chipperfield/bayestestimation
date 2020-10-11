import numpy as np
import pandas as pd
import arviz as az
import pystan as pystan

from bayestestimation.model.best import model
from bayespropestimation.bayesprophelpers import _calculate_map

class BayesTEstimation:


    def __init__(self):
        self.stan_model = self._compile_model()
        par_list = ['mu_a', 
                    'mu_b', 
                    'mu_delta',
                    'sigma_a', 
                    'sigma_b', 
                    'nu']
        self.par_list = par_list
        
    def _compile_model(self):
        # Compiles the stan model in C++
        return pystan.StanModel(model_code=model, model_name='BEST')

    def _check_sample_posterior_inputs(self):
        # Checks that parameters are in the correct format
        types = ['list', 'ndarray', 'Series']
        mu_types = ['float', 'int']
        if ((type(self.a).__name__ not in types) or (type(self.b).__name__ not in types)):
            raise ValueError("type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'Series'")
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
        return np.mean(np.concatenate([self.a, self.b]))

    def _estimate_prior_s(self):
        # Estimates a prior value for s when one isn't provided
        return np.std(np.concatenate([self.a, self.b]))

    def _build_input_dictionary(self):
        # Builds a dictionary input to apply to the model
        return {'a': self.a,
                'b': self.b,
                'n_a': self.n_a,
                'n_b': self.n_b,
                'mu': self.prior_mu,
                's': self.prior_s,
                'phi': self.prior_phi,
                'alpha': self.prior_alpha,
                'beta': self.prior_beta}

    def _extract_posteriors(self):
        # Extracts samples of the posteriors from the stan model
        self.unpermuted_extract = {}
        for i in self.par_list:
            self.unpermuted_extract[i] = self.fit.extract(pars=i, permuted=False)[i]

    def _flatten_extracts(self):
        # Flattens the chains of the posteriors
        dic = {}
        for i in self.par_list:
            dic[i] = self.unpermuted_extract[i].flatten()
        return dic

    def fit_posteriors(self, 
                       a, 
                       b, 
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
        self.a = a
        self.b = b
        self.n_a = len(a)
        self.n_b = len(b)
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
        self._extract_posteriors()

    def get_posteriors(self):
        '''
        doc string
        '''
        return self._flatten_extracts()

    def _calculate_quantiles(self, d, mean, quantiles):
        # Calculate mean and quantiles
        q = np.quantile(d, quantiles)
        if mean is True:
            q = np.append(q, np.mean(d))    
        return q

    def quantile_summary(self, mean=True, quantiles=[0.025, 0.5, 0.975], names=None):
        '''
        Summarises the properties of the estimated posterior using quantiles
        Parameters
        ----------
        mean:  boolean, default True, calculates the mean of the draws from the posterior.  Default True
        quantiles: list, calculates the quantiles of the draws from the posterior.  Default [0.025, 0.5, 0.975]
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        pd.DataFrame:  
            'theta_a':  summaries of the posterior of theta_a
            'theta_b':  summaries of the posterior of theta_b
            'delta':  summaries of the posterior of theta_b - theta_a
        '''
        if quantiles is None:
            raise ValueError("quantiles must be a list of length > 0")
        draws = list(self._flatten_extracts().values())      
        if names is None:
            names = self.par_list
        if len(names) != 6:
            raise ValueError('names must be a list of length 6')
        q = []
        for i in draws:
            q.append(self._calculate_quantiles(i, mean, quantiles))
        df = pd.DataFrame(np.array(q))
        if mean is True:
            df.columns = list(map(str, quantiles)) + ['mean']
        else:
            df.columns = list(map(str, quantiles)) 
        df['parameter'] = names
        return df
        
    def _calculate_hdi_and_map(self, d, mean, interval):
        # Calculate HDI interval and MAP
        q = az.hdi(d, hdi_prob=interval)
        m = _calculate_map(d)
        q = np.array([q[0], m, q[1]])
        if mean is True:
            q = np.append(q, np.mean(d))
        return q

    def hdi_summary(self, mean=True, interval=0.95, names=None):
        '''
        Summarises the properties of the estimated posterior using the MAP and HDI
        Parameters
        ----------
        mean:  boolean, calculates the mean of the draws from the posterior.  Default True
        interval: float, defines the HDI interval.  Default = 0.95 (i.e. 95% HDI interval)
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['theta_a', 'theta_b', 'delta']
        Returns
        -------
        pd.DataFrame:  
            'theta_a':  summaries of the posterior of theta_a
            'theta_b':  summaries of the posterior of theta_b
            'delta':  summaries of the posterior of theta_b - theta_a
        '''
        if interval is None or interval <= 0 or interval >= 1:
            raise ValueError("interval must be a float > 0 and < 1")      
        draws = list(self._flatten_extracts().values())   
        if names is None:
            names = self.par_list
        if len(names) != 6:
            raise ValueError('names must be a list of length 6')
        q = []
        for i in draws:
            q.append(self._calculate_hdi_and_map(i, mean, interval))        
        df = pd.DataFrame(np.array(q))
        col_names = ['%.5g' % ((1 - interval) / 2), 'MAP', '%.5g' % (interval + ((1 - interval) / 2))]
        if mean is True:
            df.columns = col_names + ['mean']
        else:
            df.columns = col_names
        df['parameter'] = names
        return df  

    def _probability_interpretation_guide(self, p):
        # Interpretation guide for probabilities using: 
        # https://www.cia.gov/library/center-for-the-study-of-intelligence/csi-publications/books-and-monographs/sherman-kent-and-the-board-of-national-estimates-collected-essays/6words.html
        if p >= 0 and p <= 0.13:
            i = 'almost certainly not'
        elif p > 0.13 and p <= 0.4:
            i = 'probably not'
        elif p > 0.4 and p <= 0.6:
            i = 'about equally likely'
        elif p > 0.6 and p <= 0.86:
            i = 'probably'
        elif p > 0.86 and p <= 1:
            i = 'almost certainly'
        else:
            raise ValueError('p must be >= 0 and <= 1')
        return i

    def _print_inference_probability(self, p, i, direction, value, names):
        # Combines inference values into a readable string
        s = 'The probability that ' + names[1] + ' is ' + direction + ' ' + names[0]
        if value != 0:
            s = s + ' by more than ' + str(value)
        s = s + ' is ' + ('%.5g' % (p * 100)) + '%.'
        s = s + ' Therefore ' + names[1] + ' is ' + i + ' ' + direction + ' ' + names[0]
        if value != 0:
            s = s + ' by more than ' + str(value)
        s = s + '.'
        return s

    def infer_delta_probability(self, direction='greater than', value=0, print_inference=True, names=None):
        '''
        Provides a guide to making inferences on the posterior delta, based on proportion of
        draws to the right or left of a given value. 
        Parameters
        ----------
        direction: str, defines the direction of the inference, options 'greater than' or 'less than'.  Default is 'greater than'.
        value: float,  defines the value about which to make the inference.  Default = 0.
        print_inference:  boolean, prints a readable string.  Default is True.
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['mu_a', 'mu_b', 'mu_delta']
        Returns
        -------
        tuple
            - float, probability that b > (a + value) or b < (a + value).
            - str, string interpretation of that probabiliyu       
        '''
        dir_opts = ['greater than', 'less than']
        if direction not in dir_opts:
            raise ValueError("direction must be 'greater than' or 'less than'")
        d = self.unpermuted_extract['mu_delta'].flatten()
        if direction == 'greater than':
            p = len(d[d > 0]) / len(d)
        else:
            p = len(d[d < 0]) / len(d)
        i = self._probability_interpretation_guide(p)
        if names is None:
            names = ['mu_a',
                     'mu_b',
                     'mu_delta']
        if len(names) != 3:
            raise ValueError('names must be a list of length 3')
        if print_inference is True:
            print(self._print_inference_probability(p, i, direction, value, names))
        return p, i