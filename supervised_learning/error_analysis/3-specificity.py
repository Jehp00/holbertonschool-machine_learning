#!/usr/bin/env python3
"""module error analysis"""
import numpy as np


def specificity(confusion):
    """calculates the specificity for each class in
    a confusion matrix"""
    classes = confusion.shape[0]
    specificity = []
    for current_class in range(classes):
        true_negative = 0
        totally = 0
        for r in range(classes):
            if r == current_class:
                continue
            for c in range(classes):
                if c != current_class:
                    true_negative += confusion[r][c]
                totally += confusion[r][c]
        specificity.append(true_negative / totally)
    return np.asarray(specificity)
