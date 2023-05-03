#!/usr/bin/env python3
"""module error analysis"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a
    confusion matrix"""
    classes = confusion.shape[0]
    precision = []
    for c in range(classes):
        correct = 0
        totally = 0
        for r in range(classes):
            if r == c:
                correct += confusion[r][c]
            totally += confusion[r][c]
        precision.append(correct / totally)
    return np.asarray(precision)
