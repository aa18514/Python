import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List
from collections import OrderedDict
from numpy import log

Vector_int = List[int]


DATE_STR = ['Jan', 'Feb', 'March', 'April',
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


def visualize_data(e_consumption: Vector_int,
                   title: str, x_label: str,
                   y_label: str)->(Vector_int, str):
    """
    @data: electricity consumption in MW
    """
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x = np.arange(0, len(e_consumption))
    plt.xticks(np.arange(min(x), max(x) + 1, 1500), DATE_STR)
    plt.plot(x, e_consumption)


def partition_data(dt_index: Vector_int)->Vector_int:
    """
    plot vertical lines separating the data into multiple months
    :return:
    """
    for i in range(len(dt_index)):
        plt.axvline(x=dt_index[i], c='r')


def print_data_summary(data: Vector_int, dt_index: Vector_int, granularity_constant: int)->(Vector_int, int):
    mean = []
    for i in range(0, len(dt_index) - 1):
        mean.append(np.mean(data[dt_index[i]:dt_index[i+1]]))
    x = np.arange(0, len(mean))
    plt.xticks(np.arange(min(x), max(x) + 1, granularity_constant), DATE_STR)
    plt.plot(x, mean)


def pre_process(data: Vector_int, interval: int)->(Vector_int, int):
    data = remove_seasonality(data)
    for i in range(interval):
        data = remove_trend(data)
    return data


def decode_electricity_data(path: str, net_demand: str, settlement_date: str)->str:
    dt_index = load_file_attributes(path, net_demand, settlement_date)
    dt_index = np.sort(dt_index)
    dt_index = np.where(dt_index[:-1] != dt_index[1:])[0]
    data, dt_sorted = query_data(2017)
    visualize_data(data,
                   'time series - original',
                   'time in 2017',
                   'original features')
    dt_index = np.append(0, dt_index)
    dt_index = np.append(dt_index, len(data) - 1)
    partition_data(dt_index)
    data = pre_process(data, 2)
    visualize_data(data,
                   'de-trended and de-seasonalized time series',
                   'time in 2017',
                   'modified features')
    partition_data(dt_index)
    plt.show()


if __name__ == "__main__":
    decode_electricity_data("DemandData_2017.csv", 'ND', 'SETTLEMENT_DATE')
