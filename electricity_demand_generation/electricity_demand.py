import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List
from collections import OrderedDict
from numpy import log
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from matplotlib.axes import Axes

Vector_int = List[int]
Vector_float = List[float]

DATE_STR = ['Jan', 'Feb', 'Mar', 'Apr',
            'May', 'Jun', 'Jul', 'Aug',
            'Sept', 'Oct', 'Nov', 'Dec']

ELECTRICITY_YEARS = {
        2017: {},
        2016: {},
        2015: {},
        2014: {},
        2013: {},
        2012: {},
        2011: {}
    }


def remove_seasonality(data: Vector_int)->Vector_int:
    return log(data)


def remove_trend(x_data: Vector_int)->Vector_int:
    """
    :param x_data: time series data
     first-order differencing to remove trend
    :return: de-trended series
    """
    diff = list()
    for i in range(1, len(x_data)):
        value = x_data[i] - x_data[i - 1]
        diff.append(value)
    return diff


def load_file_attributes(path: str, nd_key: str, settlement_key: str) -> str:
    """return electricity consumption in MW,
    and the settlement dates"""
    reader = csv.DictReader(open(path, 'r'))
    months = []
    for line in reader:
        val = datetime.strptime(line[settlement_key], '%d/%m/%Y')
        months.append(val.month)
        if val not in ELECTRICITY_YEARS[val.year]:
                ELECTRICITY_YEARS[val.year][val] = [int(line[nd_key])]
        else:
                ELECTRICITY_YEARS[val.year][val].append(int(line[nd_key]))
    return np.array(months)


def query_data(year: int)->int:
    ordered_dict = OrderedDict(sorted(ELECTRICITY_YEARS[year].items(),
                                      key=lambda t: t[0]))
    dict_list = []
    data = list(ordered_dict.values())
    for sublist in data:
        for val in sublist:
            dict_list.append(val)
    dict_list = np.array(dict_list, dtype=int)
    dates = np.array(list(ordered_dict.keys()))
    return dict_list, dates


def visualize_data(ax: Axes, y: Vector_int,
                   title: str, x_label: str,
                   y_label: str,
                   label: str)->(Axes, Vector_int, str):
    """
    @data: electricity consumption in MW
    """
    x = np.arange(0, len(y))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(np.arange(min(x), max(x), 1500))
    ax.set_xticklabels(DATE_STR)
    ax.plot(x, y, label=label)
    ax.legend(loc='lower left')


def partition_data(dt_index: Vector_int, ax)->Vector_int:
    """
    plot vertical lines separating the data into multiple months
    :return:
    """
    for i in range(len(dt_index)):
        if ax is None:
            plt.axvline(x=dt_index[i], c='r')
        else:
            ax.axvline(x=dt_index[i], c='r')


def print_data_summary(data: Vector_int, dt_index: Vector_int, granularity_constant: int)->(Vector_int, int):
    mean = []
    for i in range(0, len(dt_index) - 1):
        mean.append(np.mean(data[dt_index[i]:dt_index[i+1]]))
    x = np.arange(0, len(mean))
    plt.xticks(np.arange(min(x), max(x) + 1, granularity_constant), DATE_STR)
    plt.plot(x, mean)


def pre_process(data: Vector_int, interval: int)->(Vector_int, int):
    data_diff = None
    data = remove_seasonality(data)
    for i in range(interval):
        data_diff = remove_trend(data)
    return data_diff, data


def dickey_fuller_test(series: Vector_int)->Vector_int:
    """
    :param series: time series data
    :return: p value, a p-value less than 0.05 indicates that the hypothesis H0,
    i.e the series is stationary should be accepted
    """
    s_test = adfuller(series, autolag='AIC')
    return s_test[1]


def auto_correlation_test(series: Vector_float)->Vector_float:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    #ACF chart
    plot_acf(series, lags=20, ax=ax1)
    #draw 95% confidence interval line
    ax1.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    ax1.set_xlabel('Lags')
    plot_pacf(series, lags=20, ax=ax2)
    ax2.axhline(y=-1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(series)), linestyle='--', color='gray')
    ax2.set_xlabel('Lags')
    plt.show()


def ARIMA_rolling_forecast(series: Vector_float)->Vector_float:
    streaming_data_model = list()
    predictions = list()
    observations = list()
    low_ci = list()
    upper_ci = list()
    for i in range(2500):
        streaming_data_model.append(series[i])
    for i in range(2500, 2600):
        model = ARIMA(streaming_data_model, order=(1, 0, 1))
        model_fit = model.fit(disp=0)
        predictions.append(model_fit.forecast()[0])
        observations.append(series[i])
        streaming_data_model.append(series[i])
        l, h = model_fit.forecast()[2][0]
        low_ci.append(l)
        upper_ci.append(h)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('predicted vs observed electricity net demand')
    plt.xlabel('observations')
    plt.ylabel('net demand for electricity/KW')
    x = np.arange(0, len(observations))
    plt.plot(x, np.exp(observations), label='observation')
    plt.plot(x, np.exp(predictions).flatten(), label='prediction')
    ax.fill_between(x, np.exp(low_ci), np.exp(upper_ci), color='#539caf', alpha=0.4, label='95% CI')
    ax.legend(loc='lower left')
    plt.show()


def build_arima_model(series: Vector_float, dt_index)->(Vector_float, Vector_int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    model = ARIMA(series, order=(1, 0, 1))
    results_ARIMA = model.fit(disp=0)
    prediction_values = results_ARIMA.predict()
    visualize_data(ax1, prediction_values, 'predicted_time_series', 'time',
                   'consumption in KW', None)
    partition_data(dt_index, ax1)
    visualize_data(ax2, prediction_values, 'original time series', 'time',
                   'consumption in KW', None)
    partition_data(dt_index, ax2)
    plt.show()


def decode_electricity_data(path: str, net_demand: str, settlement_date: str)->str:
    dt_index = load_file_attributes(path, net_demand, settlement_date)
    dt_index = np.sort(dt_index)
    dt_index = np.where(dt_index[:-1] != dt_index[1:])[0]
    data, dt_sorted = query_data(2017)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    visualize_data(ax, data,
                   'NATIONAL GRID time series - original',
                   'time in 2017',
                   'electricity consumption in MW',
                   'original features')
    dt_index = np.append(0, dt_index)
    dt_index = np.append(dt_index, len(data) - 1)
    partition_data(dt_index, None)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_diff, data = pre_process(data, 1)
    visualize_data(ax, data,
                   'NATIONAL GRID de-trended and de-seasonalized time series',
                   'time in 2017',
                   'modified features',
                   'sixth-order log differencing')
    partition_data(dt_index, None)
    plt.show()
    p_val = dickey_fuller_test(data)
    auto_correlation_test(data)
    ARIMA_rolling_forecast(data)
    build_arima_model(data, dt_index)
    print(p_val)


if __name__ == "__main__":
    decode_electricity_data("DemandData_2017.csv", 'ND', 'SETTLEMENT_DATE')
