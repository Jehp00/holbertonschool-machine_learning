#!/usr/bin/env python3
"""module  optimization"""


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:

    :param data: list of data to calculate the moving average of
    :param beta: weight used for the moving average
    :return: list containing the moving averages of data
    """
    av = 0
    EMA = []
    for i in range(len(data)):
        av = ((av * beta) + ((1 - beta) * data[i]))
        EMA.append(av / (1 - (beta ** (i + 1))))
    return EMA
