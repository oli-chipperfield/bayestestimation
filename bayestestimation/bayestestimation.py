import numpy as np
import pandas as pd
import arviz as az
import pystan as pystan

from plotly.subplots import make_subplots
from bayestestimation.model.best import model
from bayestestimation.bayesthelpers import _calculate_map
from bayestestimation.bayestplotters import _get_centre_lines
from bayestestimation.bayestplotters import _get_intervals 
from bayestestimation.bayestplotters import _make_density_go 
from bayestestimation.bayestplotters import _make_histogram_go 
from bayestestimation.bayestplotters import _make_area_go
from bayestestimation.bayestplotters import _make_line_go
from bayestestimation.bayestplotters import _make_delta_line

class BayesTEstimation:


    def __init__(self):
        '''
        Initialises the BayesTEstimation class and compiles the stan model
        '''   
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
        mus_types = ['float', 'int']
        if ((type(self.a).__name__ not in types) or (type(self.b).__name__ not in types)):
            raise ValueError("type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'Series'")
        if self.n is not None:
            if self.n <= 0 or str(self.n).isdigit() == False:
                raise ValueError("n must be a positive integer")
        if ((self.prior_alpha <= 0) or (self.prior_beta <= 0) or (self.prior_phi <= 0)):
            raise ValueError("the prior_alpha and/or prior_beta and/or prior_phi parameters must be > 0")
        if self.prior_mu is not None and (type(self.prior_mu).__name__ not in mus_types):
            raise ValueError("prior_mu must be None or type(prior_mu).__name__ must be 'float' or 'int'")       
        if self.prior_s is not None:
            if (type(self.prior_s).__name__ not in mus_types): 
                raise ValueError("prior_s must be None or must be > 0")
            elif self.prior_s <= 0:
                raise ValueError("prior_s must be None or must be > 0")
        if self.seed is not None:
            if str(self.seed).isdigit() == False:
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
        Fits the data and prior parameters and samples from the posterior distribution.
        Parameters
        ----------
        a: list, ndarray or Series, array of continuous data containing the results from sample a.
        b: list, ndarray or Series, array of continuous data containing the results from sample b.
        n: int, the total posterior number of posterior samples to take (after burn-in).  Default 10000.
        prior_alpha: float > 0, the alpha parameter for the inv-gamma prior distribution of the standard deviation of a and b.  Default = 0.001.
        prior_beta: float > 0, the beta parameter for the inv-gamma prior distribution of the standard deviation of a and b.  Default = 0.001.
        prior_phi: float > 0, the phi parameter for the nu prior.  Default = (1/30).
        prior_mu: float, the mean parameter for the mu priors for a and b.  Default is None.  If None the combined sample mean is used.
        prior_s: float, the standard deviation parameter for the mu priors for a and b.  Default is None.  If None the combined standard deviation is used.
        seed: integer > 0, set random seed at the start of sampling, default = None
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

    def _check_for_fit(self):
        # Checks to see if the self.fit exists
        if 'fit' not in list(self.__dict__.keys()):
            raise NameError('You must add data using the fit_posteriors method before using this method')

    def get_posteriors(self):
        '''
        Retrieves random draws from the posteriors
        Returns
        -------
        dictionary:
            - mu_a: draws from the posterior of mu_a.
            - mu_b: draws from the posterior of mu_b.
            - mu_delta: draws from the posterior of mu_delta.
            - sigma_a: draws from the posterior of sigma_a.
            - sigma_b: draws from the posterior of sigma_b.
            - nu: draws from the posterior of nu.
        '''
        self._check_for_fit()
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
        names:  list of length 6, parameter names in order: a, b, b-a.  Default ['mu_a', 'mu_b', 'mu_delta', 'sigma_a', 'sigma_b', 'nu']
        Returns
        -------
        pd.DataFrame:  
            'mu_a':  summaries of the posterior of mu_a
            'mu_b':  summaries of the posterior of mu_b
            'mu_delta':  summaries of the posterior of mu_b - mu_a
            'sigma_a': Summaries of the posterior of sigma_a
            'sigma_b':  Summaries of the posterior of sigma_b
            'nu':  Summaries of the posterior of nu.

        '''
        self._check_for_fit()
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
        Summarises the properties of the estimated posteriors using the MAP and HDI
        Parameters
        ----------
        mean:  boolean, calculates the mean of the draws from the posterior.  Default True
        interval: float, defines the HDI interval.  Default = 0.95 (i.e. 95% HDI interval)
        names:  list of length 6, parameter names in order: a, b, b-a.  Default ['mu_a', 'mu_b', 'mu_delta', 'sigma_a', 'sigma_b', 'nu']
        Returns
        -------
        pd.DataFrame:  
            'mu_a': Summaries of the posterior of mu_a
            'mu_b': Summaries of the posterior of mu_b
            'mu_delta': Summaries of the posterior of mu_b - mu_a
            'sigma_a': Summaries of the posterior of sigma_a
            'sigma_b': Summaries of the posterior of sigma_b
            'nu':  Summaries of the posterior of nu.
        '''
        self._check_for_fit()
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
            - str, string interpretation of that probability       
        '''
        self._check_for_fit()
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

    def _bayes_factor_interpretation_guide(self, bf):
        # Interpretation guide for bayes factors using: 
        # Jeffreys guide (https://en.wikipedia.org/wiki/Bayes_factor#cite_note-9)
        if np.isinf(bf) or bf > np.power(10, 2):
            i = 'decisive'
        elif bf > np.power(10, 3/2) and bf <= np.power(10, 2):
            i = 'very strong'
        elif bf > 10 and bf <= np.power(10, 3/2):
            i = 'strong'
        elif bf > np.power(10, 1/2) and bf <= 10:
            i = 'substantial'
        elif bf >= 1 and bf <= np.power(10, 1/2):
            i = 'barely worth mentioning'
        elif bf < 1:
            i = 'negative'
        else:
            raise ValueError('bf did not satisfy range of criteria')
        return i

    def _print_inference_bayes_factor(self, bf, i, direction, value, names):
        s = 'The calculated bayes factor for the hypothesis that ' + names[1] + ' is ' + direction + ' ' + names[0]
        if value != 0:
            s = s + ' by more than ' + str(value)
        s = s + ' versus the hypothesis that ' + names[0] + ' is ' + direction + ' ' + names[0]
        if value != 0:
            s = s + ' by more than ' + str(value)
        s = s + ' is ' 
        if np.isinf(bf) is True:
            s = s + 'more than 100'
        else:
            s = s + ('%.5g' % bf)
        s = s + '. Therefore the strength of evidence for this hypothesis is ' + i + '.' 
        return s

    def _estimate_bayes_factor(self, p_h1, p_h2):
        # Estimates bayes Factor
        if p_h2 == 0:
            k = np.Infinity
        else:
            k = p_h1 / p_h2
        return k

    def infer_delta_bayes_factor(self, direction='greater than', value=0, print_inference=True, names=None):
        '''
        Provides a guide to making inferences on the posterior delta, based on the Bayes Factor by estimating
        P(D|H1) / P(D|H2) for the hypotheses H1: b>(a + value) vs H2: (a + value)>b (or vice versa).  
        Where D denotes the observed data.
        Parameters
        ----------
        direction: str, defines the direction of the inference, options 'greater than' or 'less than'.  Default is 'greater than'.
        value: float,  defines the value about which to make the inference.  Default = 0.
        print_inference:  boolean, prints a readable string.  Default is True.
        names:  list of length 3, parameter names in order: a, b, b-a.  Default ['mu_a', 'mu_b', 'mu_delta']
        Returns
        -------
        tuple
            - float, bayes factor for P(D|H1) / P(D|H2) for the hypotheses H1: b>(a + value) vs H2: (a + value)>b (or vice versa).
            - str, string interpretation of that bayes factor    
        '''
        self._check_for_fit()
        dir_opts = ['greater than', 'less than']
        if direction not in dir_opts:
            raise ValueError("direction must be 'greater than' or 'less than'")
        d = self.unpermuted_extract['mu_delta'].flatten() 
        if direction == 'greater than':
            p_h1 = len(d[d> value]) / len(d)       
            p_h2 = 1 - p_h1
            bf = self._estimate_bayes_factor(p_h1, p_h2)
        else:
            p_h1 = len(d[d < value]) / len(d)       
            p_h2 = 1 - p_h1
            bf = self._estimate_bayes_factor(p_h1, p_h2)
        i = self._bayes_factor_interpretation_guide(bf)
        if names is None:
            names = [
                    'mu_a',
                    'mu_b',
                    'mu_delta'
                    ]
        if len(names) > 3:
            raise ValueError('names must be a list of length 3')
        if print_inference is True:
            print(self._print_inference_bayes_factor(bf, i, direction, value, names))
        return bf, i

    def posterior_plot(self, 
                       method='hdi', 
                       delta_line=0,
                       col='#1f77b4', 
                       bounds=None,
                       names=None,
                       fig_size=None):
        '''
        Plots the density of the draws from the posterior distribution
        Parameters
        ----------
        method: str, defines method for interval estimate and central tendency.  Default = 'hdi'
            - 'hdi':  Uses HDI and maximum aposteriori
            - 'quantile': Uses credible intervals and median
        delta_line: float, position of the vertical line on the delta plot
        col:  str, colour of plots.  Default = '#1f77b4' (muted-blue)
        bounds:  float or list, defines the boundaries of the interval
            - if method = 'hdi': float, defines the interval of the HDI. Default = 0.95
            - if method = 'quantile': list, defines the credible interval.  Default = [0.025, 0.975]
        names: list of length 3, parameter names for the plot.  Default ['theta_a', 'theta_b', 'delta']
        fig_size:  tuple(width, height), dimensions of plot.  Default is None
        '''
        self._check_for_fit()
        valid_methods = ['hdi', 'quantile']
        if method not in valid_methods:
            raise ValueError("method must be 'hdi' or 'quantile'")
        if method == 'hdi' and bounds is None:
            bounds = 0.95
        if method == 'quantile' and bounds is None:
            bounds = [0.025, 0.975]
        if method == 'hdi' and (bounds <= 0 or bounds >= 1):
            raise ValueError("if method is 'hdi' then bounds must be a float between 0 and 1")
        if method == 'quantiles' and len(bounds) != 2:
            raise ValueError("quantiles must be a list of length 2")
        if names is None:
            names = self.par_list
        if len(names) != 6:
            raise ValueError('names must be a list of length 6')   
        if method == 'hdi':
            interval_name = 'hdi'
            centre_line_name = 'map'
        else: 
            interval_name = 'credible interval'
            centre_line_name = 'median'               
        fig = make_subplots(rows=2,
                            cols=3,
                            shared_xaxes=False,
                            shared_yaxes=False,
                            subplot_titles=tuple(names))
        draws = list(self._flatten_extracts().values())
        sp_coords = np.column_stack([np.tile([1, 2, 3], 2), np.repeat([1, 2], 3)])      
        for i in range(0, 6):
            x = sp_coords[i][0]
            y = sp_coords[i][1]    
            cl = _get_centre_lines(draws[i], method=method)
            intervals = _get_intervals(draws[i], method=method, bounds=bounds)
            fig.add_trace(_make_density_go(draws[i], name='posterior density', col=col),  y, x)
            fig.add_trace(_make_histogram_go(draws[i], name='posterior draws', col=col), y, x)  
            fig.add_trace(_make_line_go(cl, name=centre_line_name, col=col), y, x)   
            fig.add_trace(_make_area_go(intervals, name=interval_name, col=col), y, x)       
        fig.update_layout(shapes=[_make_delta_line(self._flatten_extracts()['mu_delta'], delta_line=delta_line)])
        fig.update_yaxes(title_text='density', row=1, col=1)
        name_set = set()
        fig.for_each_trace(lambda trace: 
            trace.update(showlegend=False)
                if (trace.name in name_set) else name_set.add(trace.name))
        if fig_size is not None:
            fig.update_layout(height=fig_size[1], width=fig_size[0])
        return fig

    def get_rhat(self):
        '''
        Extracts and summarises the rhat convergence statistics for each parameter.
        Returns
        -------
        pd.DataFrame:  
            'parameters: parameter names as used in model fitting.
            'rhat': rhat statistics for each parameter.
        '''
        self._check_for_fit()
        fit_summary = self.fit.summary()
        rhat_index = fit_summary ['summary_colnames'].index("Rhat")
        return pd.DataFrame({'parameters': fit_summary['summary_rownames'],
                             'rhat': fit_summary["summary"][:, rhat_index]})