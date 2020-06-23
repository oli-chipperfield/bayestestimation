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




