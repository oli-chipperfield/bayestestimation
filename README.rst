============================
Bayesian estimation of means
============================


.. image:: https://img.shields.io/pypi/v/bayestestimation.svg
        :target: https://pypi.python.org/pypi/bayestestimation

Class method for Bayesian estimation and comparison of means

* Free software: MIT license

Features
--------

* Class method that acts as a wrapper for a `pystan <https://pystan.readthedocs.io/en/latest/index.html>` formulation of a Bayesian 't-test' using the `BEST <https://pubmed.ncbi.nlm.nih.gov/22774788/>` implementation.
* Estimates the posterior distributions of the mean and standard deviation parameters for two samples, A and B
* Estimates of the posterior distribution of the difference in mean parameters for two samples, A and B.
* Provides summary statistics and visualisations for the estimated parameters.
* The prior parameters, sample count, random seed, credible intervals, HDI and parameter names can all be customised.
* The stan model object and stan model fit object can be accessed just like using `pystan <https://pystan.readthedocs.io/en/latest/index.html>` directly


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

See (placeholder) details.

=====
Usage
=====

To use bayestestimation in a project::

    import bayestestimation

Simple example
--------------

(placeholder)

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

.. highlight:: shell