#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: 李珂宇
import numpy as np
from functools import wraps
import pandas as pd
import matplotlib.pyplot as plt


def input_salary(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        sal = input('请输入年薪: ')
        return f(sal, *args, **kwargs)
    return decorated


@input_salary
def tax_calculator(salary):
    real_salary = float(salary)
    point_list = [0.0,
                  36000.0,
                  144000.0,
                  300000.0,
                  420000.0,
                  660000.0,
                  960000.0]
    tax_rate_list = [3.0,
                     10.0,
                     20.0,
                     25.0,
                     30.0,
                     35.0,
                     45.0]
    level = 0
    for i in range(len(point_list)):
        if real_salary > point_list[i]:
            level = i + 1
        else:
            break
    print('level = ', level)
    cum_tax = np.sum([(point_list[i]-point_list[i-1])*tax_rate_list[i-1]*0.01 if level > 1 else 0.0 for i in range(1, level)])
    final_tax = (real_salary-point_list[level-1])*tax_rate_list[level-1]*0.01+cum_tax
    return final_tax


def draw_sth(delta=0.025):
    xrange = np.arange(-2, 2, delta)
    yrange = np.arange(-2, 2, delta)
    X, Y = np.meshgrid(xrange, yrange)
    # F is one side of the equation, G is the other
    F = 0
    G = (X**2 + Y**2 - 1)**3 - X**2 * Y**3
    plt.contour((F - G), [0])
    plt.show()


def lambda_sort(array=None, reverse=False):
    if array is None:
        array = [{"name": "jack", "age": 50}, {"name": "henry", "age": 30}, {"name": "jane", "age": 25},
                 {"name": "mary", "age": 35}]
    new_array = sorted(array, key=lambda x: x["age"], reverse=reverse)
    return new_array


def euclidean_metric(array_x=None, array_y=None):
    if array_y is None:
        array_y = [7, 2, 10, 2, 7, 4, 9, 4, 9, 8]
    if array_x is None:
        array_x = [1, 2, 3, 2, 3, 4, 3, 4, 5, 6]
    return np.sqrt(np.sum((np.array(array_x)-np.array(array_y))**2)) if len(array_x) == len(array_y) else None


def c_degree_f():
    dr = pd.date_range(start='2019-01-01', periods=5)
    df = pd.DataFrame({"c": [20, 21, 15, 22, 19]}, index=dr)
    df['f'] = df.loc[:, 'c'].to_numpy()*1.8+32
    # 𝐹=𝐶 ×1.8 +32
    return df


# 1
print(tax_calculator())
# 3
print(lambda_sort())
# 4
print(euclidean_metric())
# 5
print(c_degree_f())
# 2
draw_sth()
