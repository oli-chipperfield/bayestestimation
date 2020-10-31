#!/usr/bin/env python
import pytest
import numpy as np
import pandas as pd
from bayestestimation.bayestestimation import BayesTEstimation
from bayestestimation.bayesthelpers import _calculate_kde
from bayestestimation.bayesthelpers import _calculate_map
from bayestestimation.bayestplotters import _get_centre_lines
from bayestestimation.bayestplotters import _get_intervals 
from bayestestimation.bayestplotters import _make_density_go 
from bayestestimation.bayestplotters import _make_histogram_go 
from bayestestimation.bayestplotters import _make_area_go
from bayestestimation.bayestplotters import _make_line_go
from bayestestimation.bayestplotters import _make_delta_line


def compare_dictionaries(p, z):
    k = list(z.keys())
    r = []
    for i in k:
        r.append(np.any(p[i], z[i]))
    return np.any(r)

def make_random_numpy(seed, size):
    np.random.seed(seed)
    return np.random.normal(size=size)

# Define fit_posteriors fixtures

@pytest.fixture
def make_a_numpy():
    return make_random_numpy(2020, 20)

@pytest.fixture
def make_a_list():
    return list(make_random_numpy(2020, 20))

@pytest.fixture
def make_a_series():
    return pd.DataFrame({'a': make_random_numpy(2020, 20)})['a']

@pytest.fixture
def make_b_numpy():
    return make_random_numpy(2021, 21)

@pytest.fixture
def make_b_list():
    return list(make_random_numpy(2021, 21))

@pytest.fixture
def make_b_series():
    return pd.DataFrame({'a': make_random_numpy(2021, 21)})['a']

@pytest.fixture
def make_explicit_n():
    return 5

@pytest.fixture
def make_explicit_prior():
    return 1

@pytest.fixture
def make_explicit_seed():
    return 1000

# Define fit_posteriors bad fixtures

@pytest.fixture
def make_a_str():
    return 'foo'

@pytest.fixture
def make_b_str():
    return 'bar'

@pytest.fixture
def make_invalid_n():
    return -1

@pytest.fixture
def make_invalid_prior_alpha_beta():
    return -1

@pytest.fixture
def make_invalid_prior_phi():
    return -1

@pytest.fixture
def make_invalid_prior_mu():
    return 'foobar'

@pytest.fixture
def make_invalid_prior_s_1():
    return 'bar'

@pytest.fixture
def make_invalid_prior_s_2():
    return -1

@pytest.fixture
def make_invalid_seed():
    return 1.2    

# Define fixtures for bayeprophelpers testing

@pytest.fixture
def make_draw():
    np.random.seed(1000)
    return np.random.beta(2, 5, 100)

@pytest.fixture
def _calculate_kde_results():
    return (np.array([0.03857532, 0.43082818, 0.82308105]),
            np.array([1.14415602, 1.12407264, 0.11679198]))

@pytest.fixture
def _calculate_map_results():
    return 0.038575315797957296

# Run helper tests

def test__calculate_kde_returns_correct_values(make_draw, _calculate_kde_results):
    x, kde_density = _calculate_kde(make_draw, num=3)
    assert np.allclose(x, _calculate_kde_results[0])
    assert np.allclose(kde_density, _calculate_kde_results[1])

def test__calculate_map_results(make_draw, _calculate_map_results):
    assert np.isclose(_calculate_map_results, _calculate_map(make_draw, num = 3))

# Run initialisation test

def test_BayesTEstimation_initialises():
    try:
        BayesTEstimation()
    except:
        raise pytest.fail()

# Run fit_posteriors handles bad inputs

def test_BayesTEstimattion_handles_inputs_correctly(make_a_numpy,
                                                    make_b_numpy,
                                                    make_a_str,
                                                    make_b_str,
                                                    make_invalid_n,
                                                    make_invalid_prior_alpha_beta,
                                                    make_invalid_prior_phi,
                                                    make_invalid_prior_mu,
                                                    make_invalid_prior_s_1,
                                                    make_invalid_prior_s_2,
                                                    make_invalid_seed):                                             
    # Initialisation
    BayesTEstimationClass = BayesTEstimation()
    # Good inputs
    

    # Bad inputs
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_str, b=make_b_numpy)
    assert str(e.value) == "type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'Series'"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_str)
    assert str(e.value) == "type(a).__name__ and/or type(b).__name__ must be 'list', 'ndarray' or 'Series'"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, n=make_invalid_n)
    assert str(e.value) == "n must be a positive integer"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_alpha=make_invalid_prior_alpha_beta)   
    assert str(e.value) == "the prior_alpha and/or prior_beta and/or prior_phi parameters must be > 0"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_beta=make_invalid_prior_alpha_beta)   
    assert str(e.value) == "the prior_alpha and/or prior_beta and/or prior_phi parameters must be > 0"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_phi=make_invalid_prior_phi)   
    assert str(e.value) == "the prior_alpha and/or prior_beta and/or prior_phi parameters must be > 0"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_mu=make_invalid_prior_mu)   
    assert str(e.value) == "prior_mu must be None or type(prior_mu).__name__ must be 'float' or 'int'"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_s=make_invalid_prior_s_1)   
    assert str(e.value) == "prior_s must be None or must be > 0"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, prior_s=make_invalid_prior_s_2)   
    assert str(e.value) == "prior_s must be None or must be > 0"
    with pytest.raises(ValueError) as e:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy, seed=make_invalid_seed)   
    assert str(e.value) == "seed must be a positive integer or None"
