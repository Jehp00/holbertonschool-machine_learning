#!/usr/bin/env python3
"""module error analysis"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculates the F1 score for each class
    in a confusion matrix"""
    pre = precision(confusion)
    sen = sensitivity(confusion)
    F1 = (2 * pre * sen) / (pre + sen)
    return F1
