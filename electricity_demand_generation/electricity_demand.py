import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import List
Vector_int = List[int]


DATE_STR = ['Jan', 'Feb', 'March', 'April',
            'May', 'Jun', 'Jul', 'Aug',
            'Sept', 'Oct', 'Nov', 'Dec']


def return_file_attributes(path: str, key: str) -> str:
    """return electricity consumption in MW,
    and the settlement dates"""
    reader = csv.DictReader(open(path, 'r'))
    dict_list = []
    dates = []
    for line in reader:
        val = datetime.strptime(line['SETTLEMENT_DATE'], '%d/%m/%Y')
        if val.year == 2017:
            dict_list.append((line[key]))
            dates.append(val.month)
    dict_list = np.array(dict_list, dtype=int)
    dict_list = dict_list[np.argsort(dates)]
    dates = np.array(dates)
    dates = np.sort(dates)
    return dict_list, dates


def visualize_data(e_consumption: Vector_int)->Vector_int:
    """
    @data: electricity consumption in MW
    """
    plt.xlabel('time in 2017')
    plt.ylabel('electricity consumption/MW')
    x = np.arange(0, len(e_consumption))
    plt.figure(1)
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
    plt.figure(2)
    plt.xticks(np.arange(min(x), max(x) + 1, granularity_constant), DATE_STR)
    plt.plot(x, mean)


def decode_electricity_data(path: str, net_demand: str)->str:
    data, dt_sorted = return_file_attributes(path, net_demand)
    dt_index = np.where(dt_sorted[:-1] != dt_sorted[1:])[0] + 1
    dt_index = np.append(0, dt_index)
    dt_index = np.append(dt_index, len(data) - 1)
    print_data_summary(data, dt_index, 1)
    plt.show()


if __name__ == "__main__":
    decode_electricity_data("DemandData_2017.csv", 'ND')