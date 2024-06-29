#!/usr/bin/env python3
""" This module contains the class Yolo
    that implements the You Only Look Once Algorithm

    requires:
        - Tensorflow
        - Numpy
        - yolo.hs in the working directory
"""

import tensorflow.keras as K
import numpy as np


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

    def process_outputs(self, outputs, image_size):
        """ Processes the output
            - outputs: list, numpy.ndarray's containing the predictions
                       from the Darknet model for a single image:
                - (grid_height, grid_width, anchor_boxes, 4 + 1 + classes);
                    - grid_height & grid_width: height and width
                                                of the grid used for the output
                    - anchor_boxes: number of anchor boxes used
                    - 4: (t_x, t_y, t_w, t_h)
                    - 1: Objectness
                    - classes: class probabilities for all classes
            - image_size: numpy.ndarray, contains the image's original size [image_height, image_width]
        """

        def sigmoid(x):
            """ Performs sigmoid calculation over an input X """
            return 1 / (1 + np.exp(-x))

        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        for i, output in enumerate(outputs):
            gh = output.shape[0]
            gw = output.shape[1]

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            anchor = self.anchors[i]

            pw = anchor[:, 0]
            ph = anchor[:, 1]
            pw = pw.reshape(1, 1, len(pw))
            ph = ph.reshape(1, 1, len(ph))

            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gw),
                         gh).reshape(gh, gh, 1).T.reshape(gh, gh, 1)

            bx = sigmoid(t_x) + cx
            by = sigmoid(t_y) + cy

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bx = bx / gw
            by = by / gh
            bw = bw / int(self.model.input.shape[1])
            bh = bh / int(self.model.input.shape[2])

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[:, :, :, :4].shape)
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)

            box_confidence = sigmoid(output[:, :, :, 4, np.newaxis])
            box_confidences.append(box_confidence)

            box_class = sigmoid(output[:, :, :, 5:])
            box_class_probs.append(box_class)

        return boxes, box_confidences, box_class_probs
