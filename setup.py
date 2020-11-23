#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy>=1.17.2",
    "scipy>=1.3.1",
    "pandas>=0.25.1",
    "arviz>=0.9.0",
    "pystan>=2.19.1.1",
    "plotly>=4.9.0",
]

setup_requirements = [
    "pytest-runner",
    "numpy>=1.17.2",
    "scipy>=1.3.1",
    "pandas>=0.25.1",
    "arviz>=0.9.0",
    "pystan>=2.19.1.1",
    "plotly>=4.9.0",
]

test_requirements = [
    "pytest>=3",
    "numpy>=1.17.2",
    "scipy>=1.3.1",
    "pandas>=0.25.1",
    "arviz>=0.9.0",
    "pystan>=2.19.1.1",
    "plotly>=4.9.0",
]

long_description = """

Class method for Bayesian estimation and comparison of means

* Free software: MIT license

Features
--------

Class method that acts as a wrapper for a pystan formulation of a Bayesian 't-test' using the BEST implementation.

- Estimates the posterior distributions of the mean and standard deviation parameters for two samples, A and B

- Estimates of the posterior distribution of the difference in mean parameters for two samples, A and B.

- Provides summary statistics and visualisations for the estimated parameters.

- The prior parameters, sample count, random seed, credible intervals, HDI and parameter names can all be customised.

- The stan model object and stan model fit object can be accessed just like using pystan directly
"""

setup(
    author="Oliver Chipperfield",
    author_email="omc0dev@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Class method for Bayesian estimation and comparison of means",
    install_requires=requirements,
    license="MIT license",
    long_description=long_description,
    include_package_data=True,
    keywords="bayestestimation",
    name="bayestestimation",
    packages=find_packages(include=["bayestestimation", "bayestestimation.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/oli-chipperfield/bayestestimation",
    version="0.9.2",
    zip_safe=False,
)
