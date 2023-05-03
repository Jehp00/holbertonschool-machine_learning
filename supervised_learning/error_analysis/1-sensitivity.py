#!/usr/bin/env python3
"""module error analysis"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each
    class in a confusion matrix"""
    classes = confusion.shape[0]
    sensitivity = []
    for r in range(classes):
        correct = 0
        totally = 0
        for c in range(classes):
            if r == c:
                correct += confusion[r][c]
            totally += confusion[r][c]
        sensitivity.append(correct / totally)
    return np.asarray(sensitivity)
