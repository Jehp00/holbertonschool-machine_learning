#!/usr/bin/env python3
"""module tensor flow"""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation graph"""
    for i in range(len(layer_sizes)):
        if i == 0:
            lay = create_layer(x, layer_sizes[0], activations[0])
        else:
            lay = create_layer(lay, layer_sizes[i], activations[i])

        return lay
