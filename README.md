# tsforest

[![Build Status][travis-image]][travis-url]  [![PyPI version][pypi-image]][pypi-url]  [![PyPI download][download-image]][pypi-url]

Using Gradient Boosting Regression Trees (GBRT) for multiple time series forecasting problems has proven to be very effective. This package provides a complete framework for efficiently handle multiple time-series datasets and building a GBRT forecast model.

## Key Features
* Easy to use interface using [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [XGBoost](https://xgboost.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/) & [H2O GBM](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html) as backends.
* Automatic time series features enginnering:
  * Time-based attributes.
  * Fast/Parallel computation of lag features.
  * Fast/Parallel computation of rolling window statistics.
  * Definition of custom rolling window statistics.
* Fast computation of recursive one-step ahead prediction when using lagged features.
* Categorical encoding through [category-encoders](http://contrib.scikit-learn.org/category_encoders/).
* Automatic/Parallel trend removal by time serie.
* Automatic/Parallel scaling by time serie.

## Installation

Via [PyPI](https://pypi.org/project/tsforest/):

```bash
pip install tsforest
```
Or you can clone this repository and install it from source: 

```bash
python setup.py install
```

[travis-image]: https://travis-ci.org/mavillan/tsforest.svg?branch=master
[travis-url]: https://travis-ci.org/mavillan/tsforest
[pypi-image]: http://img.shields.io/pypi/v/tsforest.svg
[download-image]: http://img.shields.io/pypi/dm/tsforest.svg
[pypi-url]: https://pypi.org/project/tsforest/
