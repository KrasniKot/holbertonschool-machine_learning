#!/usr/bin/env python3
""" This module contains the class Yolo
    that implements the You Only Look Once Algorithm

    requires:
        - Tensorflow
        - yolo.hs in the working directory
"""

import tensorflow.keras as K


class Yolo:
    """ Defines the Yolo algorithm to perform object detection"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Initializes the Yolo algorithm
            - model_path: path to where a Darknet Keras model is stored
            - classes_path: path to where the list of class names used
                            for the Darknet model, listed in order of index,
                            can be found.
            - class_t: float, represents the box score threshold
                       for the initial filtering step
            - nms_t: float, represent the IOU threshold for non-max suppression
            - anchors: numpy.ndarray (outputs, anchor_boxes, 2);
                       contains all of the anchor boxes:
                        - outputs: number of outputs (predictions)
                                   made by the Darknet model
                        - anchor_boxes: number of anchor boxes used
                                        for each prediction
                        - 2:  [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
