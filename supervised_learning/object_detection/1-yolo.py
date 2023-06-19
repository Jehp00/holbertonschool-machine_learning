#!/usr/bin/env python3
"""This model contains the class Yolo"""
import tensorflow.keras as K
import numpy as np


class Yolo():
    """uses the Yolo v3 algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        :param model_path: the path to where a Darknet Keras model is stored
        :param classes_path: the path to where the list of class names used for
                      the Darknet model, listed in order of index, can be found
        :param class_t: a float representing the box score threshold for the initial
                 filtering step
        :param nms_t: a float representing the IOU threshold for non-max suppression
        :param anchors: a numpy.ndarray of shape (outputs, anchor_boxes, 2) containing
                 all of the anchor boxes:
            outputs: the number of outputs (predictions) made by the Darknet
                     model
            anchor_boxes: the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            lines = f.readlines()
            self.class_name = []
            for name in lines:
                self.class_name.append(name[:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Returns the output after passing through Sigmoid function"""
        return (1. / (1. + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """Processes the outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            grid_height, grid_width = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            sigmoid_conf = self.sigmoid(output[..., 4])
            sigmoid_prob = self.sigmoid(output[..., 5:])

            box_conf = np.expand_dims(sigmoid_conf, axis=-1)
            box_class_prob = sigmoid_prob

            box_confidences.append(box_conf)
            box_class_probs.append(box_class_prob)

            b_wh = anchors * np.exp(t_wh)
            b_wh /= self.model.inputs[0].shape.as_list()[1:3]

            grid = np.tile(np.indices((grid_width, grid_height)).T,
                           anchors.shape[0]).reshape(
                (grid_height, grid_width) + anchors.shape)

            b_xy = (self.sigmoid(t_xy) + grid) / [grid_width, grid_height]

            b_xy1 = b_xy - (b_wh / 2)
            b_xy2 = b_xy + (b_wh / 2)
            box = np.concatenate((b_xy1, b_xy2), axis=-1)
            box *= np.tile(np.flip(image_size, axis=0), 2)

            boxes.append(box)
        return (boxes, box_confidences, box_class_probs)
