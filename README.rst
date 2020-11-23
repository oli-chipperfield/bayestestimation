============================
Bayesian estimation of means
============================


.. image:: https://img.shields.io/pypi/v/bayestestimation.svg
        :target: https://pypi.python.org/pypi/bayestestimation

Class method for Bayesian estimation and comparison of means

* Free software: MIT license

Features
--------

* Class method that acts as a wrapper for a `pystan <https://pystan.readthedocs.io/en/latest/index.html>`_ formulation of a Bayesian 't-test' using the `BEST <https://pubmed.ncbi.nlm.nih.gov/22774788/>`_ implementation.
* Estimates the posterior distributions of the mean and standard deviation parameters for two samples, A and B
* Estimates of the posterior distribution of the difference in mean parameters for two samples, A and B.
* Provides summary statistics and visualisations for the estimated parameters.
* The prior parameters, sample count, random seed, credible intervals, HDI and parameter names can all be customised.
* The stan model object and stan model fit object can be accessed just like using `pystan <https://pystan.readthedocs.io/en/latest/index.html>`_ directly


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage


============
Installation
============


Stable release
--------------

To install bayestestimation, run this command in your terminal:

.. code-block:: console

    $ pip install bayestestimation

This is the preferred method to install bayestestimation, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for bayestestimation can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/oli-chipperfield/bayestestimation

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/oli-chipperfield/bayestestimation/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/oli-chipperfield/bayestestimation
.. _tarball: https://github.com/oli-chipperfield/bayestestimation/tarball/master

===========
Methodology
===========

See `notebook <hhttps://github.com/oli-chipperfield/bayestestimation/blob/master/docs/bayestestimation_basis.ipynb>` details.

=====
Usage
=====

To use bayestestimation in a project

.. code-block:: python

    import bayestestimation

Simple example
--------------

To a carry a simple estimation of the posterior density of two samples (and their delta), import the BayesTEstimation class.

.. code-block:: python

    from bayestestimation.bayestestimation import BayesTEstimation

Initialise the `BayesTEstimation` class.  Initialisation compiles the model in C++, you need only do this once.

.. code-block:: python

    ExampleBayes = BayesTEstimation()

Define data from samples A and B as two lists, numpy arrays or pandas series.

.. code-block:: python

    import numpy as np

    np.random.seed(1111)

    a = np.random.normal(0, size = 20)
    b = np.random.normal(0, size = 20)

Input the data and estimate the posterior densities using the `fit_posteriors` method.

.. code-block:: python

    ExampleBayes.fit_posteriors(a, b)

There are five methods for accessing information about the draws from simulations of the posterior densities.

.. code-block:: python

    ExampleBayes.get_posteriors()
    # Returns a dictionary of arrays of samples from the posterior distributions of parameters

.. code-block:: python

    ExampleBayes.hdi_summary()
    # Returns dataframe of the high-density-interval (HDI), maximum-a-posteriori (MAP) and mean of samples from the posteriors

.. image:: https://github.com/oli-chipperfield/bayestestimation/blob/master/images/example_hdi.png

.. code-block:: python

    ExampleBayes.quantile_summary()
    # Returns dataframe of quantiles and mean of the posterior densities of samples for parameters

.. image:: https://github.com/oli-chipperfield/bayestestimation/blob/master/images/example_quantile.png

.. code-block:: python

    ExampleBayes.infer_delta_probability()
    # Returns probability estimate of the delta parameter being greater than 0, plus an aid to inference.  
    # Includes an optional print out of the probability and inference.
    
    'The probability that mu_b is greater than mu_a is 51.13%. Therefore mu_b is about equally likely greater than mu_a.'
    '(0.5113, 'about equally likely')'

.. code-block:: python

    ExampleBayes.infer_delta_bayes_factor()
    # Returns the Bayes factor of the hypothesis that P(theta_b > theta_a | D) where D is the data, plus an aid to inference.  
    # Includes an optional print out of the Bayes factor and inference.

    'The calculated bayes factor for the hypothesis that mu_b is greater than mu_a versus the hypothesis that mu_a is greater than mu_a is 1.0462. Therefore the strength of evidence for this hypothesis is barely worth mentioning.'
    '(1.046245140167792, 'barely worth mentioning')'

.. code-block:: python

    ExampleBayes.posterior_plot()
    # Returns KDE plots of samples from the posterior densities of the parameters

.. image:: https://github.com/oli-chipperfield/bayestestimation/blob/master/images/example_posterior_plot.png

To inspect convergence, the `rhat` estimates for each parameter can be retrieved using the `get_rhat` method.

.. code-block:: python

    ExampleBayes.get_rhat()

.. image:: https://github.com/oli-chipperfield/bayestestimation/blob/master/images/example_rhat.png

To see how to use non-default parameters, refer to the `usage guid <https://github.com/oli-chipperfield/bayestestimation/blob/master/docs/bayestestimation_usage.ipynb>`_ or refer to the doc-strings in the `source <https://github.com/oli-chipperfield/bayestestimation/bayestestimation/bayestestimation.py>`_.

The `BayesTEstimation` class is a wrapper for a stan model, the model object can easily accessed and interacted with using:

.. code-block:: python

    ExampleBayes.stan_model

The `fit` object can be easily accessed using:

.. code-block:: python

    ExampleBayes.fit

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

.. highlight:: shell