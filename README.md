# Preferential Bayesian Multi-Objective Hyperparameter Optimization
[![Unittests](https://github.com/ja-thomas/pbmohpo/actions/workflows/unittests.yml/badge.svg?branch=main)](https://github.com/ja-thomas/pbmohpo/actions/workflows/unittests.yml)
[![Linting](https://github.com/ja-thomas/pbmohpo/actions/workflows/black.yml/badge.svg?branch=main)](https://github.com/ja-thomas/pbmohpo/actions/workflows/black.yml)
[![Docs](https://github.com/ja-thomas/pbmohpo/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/ja-thomas/pbmohpo/actions/workflows/docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: LGPL v2](https://img.shields.io/badge/License-LGPL_v2-blue.svg)](https://github.com/ja-thomas/pbmohpo/blob/main/LICENSE)

## Documentation

https://ja-thomas.github.io/pbmohpo/

## Summary

While hyperparameter optimization has been accepted as an important component of a machine learning task, it is often conducted in an unrealistic setting.

While research often presents Machine Learning as a one dimensional problem with a single evaluation criterion like Accuracy, real-world applications seldom present in that way:
Multiple - often conflicting - performance metrics are of interest to a decision maker (DM), thus making the decision for a fully configured model often more challenging as a suitable trade-off needs to be identified.

While this problem can be solved via expensive multi-objective black-box optimization, a DM might in reality not be able to specify all of their evaluation criteria, but simply state their preference of one model over another or produce a ranking of models given a list.

Other applications like A/B testing or recommender systems similarly only provide feedback this way.

This scenario of optimizing only through pairwise preferences has been explored in K-armed duelling bandit problems and Preferential Bayesian Optimization (PBO).
While PBO gives a direct answer to expensive black-box optimization based on pairwise preferences, the methods introduce a second bottleneck in the iterative process of Bayesian Optimization (BO).

In addition to the expensive evaluation(s) of selected models, the optimization now needs to wait for a preference expression from the DM before selecting new models to evaluate.
To avoid unnecessary idle time and create a drawn-out optimization process, mechanisms need to be developed in order to use computation resources when available as well as use DM time to rank models when the DM is available.
