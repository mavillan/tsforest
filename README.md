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

## Example Usage

```python
import pandas as pd
from tsforest.forecast import LightGBMForecaster
# could be any of: LightGBMForecast, XGBoostForecaster, CatBoostForecaster, H2OGBMForecaster

data = pd.read_csv("tests/tests_data/data_many_ts.csv", parse_dates=["ds"])
data.sample(5)
```
|     | ds                  |       y |   ts_uid |
|----:|:--------------------|--------:|---------:|
| 341 | 2018-12-08 00:00:00 | 33.6565 |        1 |
| 796 | 2018-09-08 00:00:00 | 23.179  |        2 |
| 356 | 2018-12-23 00:00:00 | 36.9956 |        1 |
| 176 | 2018-06-26 00:00:00 | 26.2164 |        1 |
| 556 | 2018-01-11 00:00:00 | 17.1385 |        2 |

Input data shoud always contain at least:
1. A column named `ds` of type `datetime64[ns]` indicating the timestamp. This sample data ranges from `2018-01-01` to `2019-06-30`.
2. A column named `y` indicating the response variable (to be predicted).
3. A list of columns to idenfity each time series (in case of multiple time series data). In the sample data is just a single column named `ts_uid`.

```python
# model settings
model_kwargs = {
    # LightGBM parameters: lightgbm.readthedocs.io/en/latest/Parameters.html
    "model_params": {
        "objective":"l2",
        "num_leaves":31,
        "learning_rate":0.1,
        "feature_fraction":0.8  
    },
    # time-attribute features
    "time_features": ["year", "month", "week_day", "month_progress", "year_week"],
    # encoding for categorical features
    "categorical_features": {"ts_uid": "default"},
    # time series unique identifier columns
    "ts_uid_columns": ["ts_uid", ],
}

# model fitting
model = LightGBMForecaster(**model_kwargs)
model.fit(train_data=data)
```
For the time-attributes you can use any from this list: [`year`, `quarter`, `month`, `days_in_month`, `year_week`, `year_day`, `month_day`, `week_day`, `hour`, `minute`, `second`, `microsecond`, `millisecond` `nanosecond`, `month_progress`, `second_cos`, `second_sin`, `minute_cos`, `minute_sin`, `hour_cos`, `hour_sin`, `week_day_cos`, `week_day_sin`, `year_day_cos`, `year_day_sin`, `year_week_cos`, `year_week_sin`, `month_cos`, `month_sin`]. Here, `*_cos` and `*_sin` correspond to the cyclical transformation of the corresponding feature.

For the categorical features, the dictionary value can be `"default"` (in this cases uses the default [categorical encoding of LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html#categorical-feature-support)) or the string name of any of the classes in [category-encoders](http://contrib.scikit-learn.org/category_encoders/).


```python
# builds the prediction dataframe
predict_data = pd.concat([pd.DataFrame({"ds":pd.date_range("2019-07-01", "2019-07-28"), "ts_uid":i}) 
                          for i in range(1,3)],
                          ignore_index=True)
predict_data.head()
```
|    | ds                  |   ts_uid |
|---:|:--------------------|---------:|
|  0 | 2019-07-01 00:00:00 |        1 |
|  1 | 2019-07-02 00:00:00 |        1 |
|  2 | 2019-07-03 00:00:00 |        1 |
|  3 | 2019-07-04 00:00:00 |        1 |
|  4 | 2019-07-05 00:00:00 |        1 |'

The prediction dataframe should contain the same columns as the input data except by `"y"`, and should contain the timestamps for the period to be predicted. 

```python
# makes predictions
forecast = model.predict(predict_data)
forecast.head()
```
|    | ds                  |   ts_uid |   y_pred |
|---:|:--------------------|---------:|---------:|
|  0 | 2019-07-01 00:00:00 |        1 |  25.0229 |
|  1 | 2019-07-02 00:00:00 |        1 |  30.3195 |
|  2 | 2019-07-03 00:00:00 |        1 |  28.9998 |
|  3 | 2019-07-04 00:00:00 |        1 |  29.121  |
|  4 | 2019-07-05 00:00:00 |        1 |  31.8027 |
