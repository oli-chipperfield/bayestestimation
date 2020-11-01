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
    return 3

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

# Define get_posteriors fixtures
@pytest.fixture
def make_get_posteriors_results():
    return np.array([[ 1.82178765,  0.26726264,  0.51371462, -0.26307566],
                     [ 0.22842242, -0.75733768,  1.51778432,  0.28337746],
                     [-1.59336522, -1.02460032,  1.0040697 ,  0.54645312],
                     [ 1.31736902,  1.19313651,  4.29328488,  0.26494139],
                     [ 0.32062644,  0.40769547,  4.28318156,  1.30801229],
                     [ 6.12807855,  3.17060426,  2.22557827,  1.55142898]])

@pytest.fixture
def make_get_posteriors_results_with_explicit_prior_alpha():
    return np.array([[ 1.82178765,  0.26726264,  0.51371462, -0.26307566],
                     [ 0.22842242, -0.75733768,  1.51778432,  0.28337746],
                     [-1.59336522, -1.02460032,  1.0040697 ,  0.54645312],
                     [ 1.30712739,  1.0530709 ,  4.16132443,  0.25679802],
                     [ 0.31813379,  0.35983497,  4.15153165,  1.2678086 ],
                     [ 6.12807855,  3.17060426,  2.22557827,  1.55142898]])

@pytest.fixture
def make_get_posteriors_results_with_explicit_prior_beta():
    return np.array([[ 1.82178765,  0.26726264,  0.51371462, -0.26307566],
                     [ 0.22842242, -0.75733768,  1.51778432,  0.28337746],
                     [-1.59336522, -1.02460032,  1.0040697 ,  0.54645312],
                     [ 1.328383  ,  1.28577873,  4.72220512,  0.30623596],
                     [ 0.33403218,  0.4156223 ,  4.30486861,  1.33270051],
                     [ 6.12807855,  3.17060426,  2.22557827,  1.55142898]])

@pytest.fixture
def make_get_posteriors_results_with_explicit_prior_phi():
    return np.array([[ 1.82178765,  0.26726264,  0.51371462, -0.26307566],
                     [ 0.22842242, -0.75733768,  1.51778432,  0.28337746],
                     [-1.59336522, -1.02460032,  1.0040697 ,  0.54645312],
                     [ 1.31736902,  1.19313651,  4.29328488,  0.26494139],
                     [ 0.32062644,  0.40769547,  4.28318156,  1.30801229],
                     [ 5.90391579,  2.76084894,  2.09393342,  1.53988395]])

@pytest.fixture
def make_get_posteriors_results_with_explicit_prior_mu():
    return np.array([[ 1.82490197,  0.31709179,  0.52617191, -0.25061837],
                     [ 0.23153675, -0.70750852,  1.53024161,  0.29583474],
                     [-1.59336522, -1.02460032,  1.0040697 ,  0.54645312],
                     [ 1.31736902,  1.19313651,  4.29328488,  0.26494139],
                     [ 0.32062644,  0.40769547,  4.28318156,  1.30801229],
                     [ 6.12807855,  3.17060426,  2.22557827,  1.55142898]])

@pytest.fixture
def make_get_posteriors_results_with_explicit_prior_s():
    return np.array([[ 1.82318499,  0.28187739,  0.51920123, -0.26547989],
                     [ 0.22860125, -0.75610507,  1.52251148,  0.2834922 ],
                     [-1.59458373, -1.03798245,  1.00331025,  0.54897209],
                     [ 1.31736902,  1.19313651,  4.29328488,  0.26494139],
                     [ 0.32062644,  0.40769547,  4.28318156,  1.30801229],
                     [ 6.12807855,  3.17060426,  2.22557827,  1.55142898]])

# Fixtures for testing results summaries

@pytest.fixture
def make_quantile_summary():
    return np.array([[-0.7514505353555653, -0.3606663387131223, 0.039108253763252415],
                     [-0.4744652899078007, -0.05335698459311282, 0.35343529956443226],
                     [-0.26205676245663007, 0.3079240594731508, 0.8664490917659036],
                     [0.6091506486807179, 0.8405139282720744, 1.2092274039739142],
                     [0.6503299704904818, 0.8887172883934797, 1.2715059996905795],
                     [5.370919434492186, 29.918093516704694, 124.19139809164547]])

@pytest.fixture
def make_hdi_summary():
    return np.array([[-0.757382144261888, -0.34234680836424614, 0.029564917758929526],
                     [-0.4555505896647165, -0.035785538079082135, 0.3699776818490415],
                     [-0.24967564966754063, 0.3219291023903855, 0.8778163616507515],
                     [0.5778856275916361, 0.8209193793911209, 1.1627010288601225],
                     [0.6247454942357719, 0.8577478205548237, 1.228128295339814],
                     [2.4371793019105104, 15.978161558029015, 103.45486599210501]])

@pytest.fixture
def make_get_rhat():
    return np.array([0.9999232374392668, 1.000098518503359, 1.000104301007721,
                     0.9997725213816185, 0.9998265674210559, 1.0001492775561391,
                     1.0006674973934064])

# Define fixtures for delta inference testing

@pytest.fixture
def make_infer_delta_probability_result():
    return (0.857, 'probably')

@pytest.fixture
def make_infer_delta_bayes_factor_result():
    return (5.9930069930069925, 'substantial')

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

def test_BayesTEstimattion_handles_initialises_and_inputs_handled_correctly(make_a_numpy,
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
    # Test good inputs
    try:
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, b=make_b_numpy)
    except:
        raise pytest.fail()
    # Test bad inputs
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

def test_BayesTEstimattion_outputs_correct_results(make_a_numpy,
                                                   make_b_numpy,
                                                   make_explicit_n,
                                                   make_explicit_prior,
                                                   make_explicit_seed,
                                                   make_get_posteriors_results,
                                                   make_get_posteriors_results_with_explicit_prior_alpha,
                                                   make_get_posteriors_results_with_explicit_prior_beta,
                                                   make_get_posteriors_results_with_explicit_prior_phi,
                                                   make_get_posteriors_results_with_explicit_prior_mu,
                                                   make_get_posteriors_results_with_explicit_prior_s):
    # Initialisation
    BayesTEstimationClass = BayesTEstimation()
    # Test outputs
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results)
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         prior_alpha=make_explicit_prior,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results_with_explicit_prior_alpha)
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         prior_beta=make_explicit_prior,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results_with_explicit_prior_beta)
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         prior_phi=make_explicit_prior,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results_with_explicit_prior_phi)
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         prior_mu=make_explicit_prior,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results_with_explicit_prior_mu)
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy,
                                         prior_s=make_explicit_prior,
                                         n=make_explicit_n,
                                         seed=make_explicit_seed)
    post = np.array(list(BayesTEstimationClass.get_posteriors().values()))
    assert np.allclose(post, make_get_posteriors_results_with_explicit_prior_s)

def test_BayesTEstimattion_outputs_correct_summaries(make_a_numpy,
                                                     make_b_numpy,
                                                     make_explicit_seed,
                                                     make_quantile_summary,
                                                     make_hdi_summary,
                                                     make_get_rhat,
                                                     make_infer_delta_probability_result,
                                                     make_infer_delta_bayes_factor_result):
    # Initialisation
    BayesTEstimationClass = BayesTEstimation()
    BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                         b=make_b_numpy, 
                                         seed=make_explicit_seed) 
    summary = np.array(BayesTEstimationClass.quantile_summary().iloc[:, [0, 1, 2]], dtype = 'float')
    assert np.allclose(summary, make_quantile_summary)
    summary = np.array(BayesTEstimationClass.hdi_summary().iloc[:, [0, 1, 2]], dtype = 'float')
    assert np.allclose(summary, make_hdi_summary)
    summary = np.array(BayesTEstimationClass.get_rhat()['rhat'])
    assert np.allclose(summary, make_get_rhat)
    p, i = BayesTEstimationClass.infer_delta_probability()
    assert np.isclose(p, make_infer_delta_probability_result[0])
    assert i == make_infer_delta_probability_result[1]
    bf, i = BayesTEstimationClass.infer_delta_bayes_factor()
    assert np.isclose(bf, make_infer_delta_bayes_factor_result[0])
    assert i == make_infer_delta_bayes_factor_result[1]

# Run bayestplotters tests

def test__get_centre_lines_runs_without_error(make_draw):
    try:
        _get_centre_lines(make_draw, method='hdi')
    except:
        raise pytest.fail()

def test__get_intervals_runs_with_hdi_without_error(make_draw):
    try:
        _get_intervals(make_draw, method='hdi', bounds=0.95)
    except:
        raise pytest.fail()

def test__get_intervals_runs_with_quantile_without_error(make_draw):
    try:
        _get_intervals(make_draw, method='quantile', bounds=[0.025, 0.975])
    except:
        raise pytest.fail()

def test__make_density_go_without_error(make_draw):
    try:
        _make_density_go(make_draw, name='dummy')
    except:
        raise pytest.fail()

def test__make_histogram_go_without_error(make_draw):
    try:
        _make_histogram_go(make_draw, name='dummy')
    except:
        raise pytest.fail()

def test__make_line_go_with_without_error(make_draw):
    try:
        cl = _get_centre_lines(make_draw, method='hdi')
        _make_line_go(cl, name='dummy')
    except:
        raise pytest.fail()

def test__make_area_go_without_error(make_draw):
    try:
        intervals = _get_intervals(make_draw, method='hdi', bounds=0.95)
        _make_area_go(intervals, name='dummy')
    except:
        raise pytest.fail()

# Run plot_posterior method test

def test_plot_posterior_without_error(make_a_numpy, make_b_numpy, make_explicit_seed):
    try:
        # Initialisation
        BayesTEstimationClass = BayesTEstimation()
        BayesTEstimationClass.fit_posteriors(a=make_a_numpy, 
                                             b=make_b_numpy,
                                             seed=make_explicit_seed)
        BayesTEstimationClass.posterior_plot()                                    
    except:
        raise pytest.fail()