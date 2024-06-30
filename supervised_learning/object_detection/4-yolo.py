#!/usr/bin/env python3
""" This module contains the class Yolo
    that implements the You Only Look Once Algorithm

    requires:
        - Tensorflow
        - Numpy
        - yolo.hs in the working directory
        - cv2
"""

import tensorflow.keras as K
import numpy as np
import glob
import cv2


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
            - image_size: numpy.ndarray, contains the image's original size;
                          [image_height, image_width]
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
            # Specify the grid's dimensions
            gh = output.shape[0]
            gw = output.shape[1]

            # Object's center position
            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]

            # Object's final predicted dimensions
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            anchor = self.anchors[i]  # Anchor for current ouput

            # Getting current anchors widths and heights
            pw = anchor[:, 0]
            ph = anchor[:, 1]
            pw = pw.reshape(1, 1, len(pw))
            ph = ph.reshape(1, 1, len(ph))

            # Getting centers for all grids
            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gw),
                         gh).reshape(gh, gh, 1).T.reshape(gh, gh, 1)

            # Calculating center of the final bounding box
            bx = sigmoid(t_x) + cx
            by = sigmoid(t_y) + cy

            # Calculating dimensions of the final bounding box
            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # Normalizing
            bx = bx / gw
            by = by / gh
            bw = bw / int(self.model.input.shape[1])
            bh = bh / int(self.model.input.shape[2])

            # Coordinates for the final bounding box
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[:, :, :, :4].shape)  # Initialized output

            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)

            # Calculating confidence
            box_confidence = sigmoid(output[:, :, :, 4, np.newaxis])
            box_confidences.append(box_confidence)

            # Calculating classes probabilities
            box_class = sigmoid(output[:, :, :, 5:])
            box_class_probs.append(box_class)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ Determines which boxes will be shown or not
                - boxes: numpy.ndarrays, of shape
                         (grid_height, grid_width, anchor_boxes, 4);
                         contains the processed boundary boxes
                         for each output, respectively
                - box_confidences: numpy.ndarrays, of shape
                                   (grid_height, grid_width, anchor_boxes, 1);
                                   contains the processed box confidences
                                   for each output, respectively
                - box_class_probs: numpy.ndarrays, of shape
                                   (grid_height,
                                   grid_width
                                   anchor_boxes,
                                   classes);
                                   contains the processed box class
                                   probabilities for each output,
                                   respectively
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, conf, prob in zip(boxes, box_confidences, box_class_probs):
            # Computing box scores
            scores = conf * prob
            max_scores = np.max(scores, axis=-1)  # Maximum score per box
            # Class with max score
            max_classes = np.argmax(scores, axis=-1)

            # Filter based on the confidence threshold
            mask = max_scores >= self.class_t

            # Apply the mask to filter boxes, classes, and scores
            filtered_boxes.append(box[mask])
            box_classes.append(max_classes[mask])
            box_scores.append(max_scores[mask])

        # Concatenate results across all outputs
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ Performs non-max suppression
            - filtered_boxes: numpy.ndarray, shape (?, 4),
                              contains all of the filtered bounding boxes
            - box_classes: numpy.ndarray, shape (?,),
                           contains the class number for the class that
                           filtered_boxes predicts, respectively
            - box_scores: numpy.ndarray, shape (?),
                          contains the box scores for each box
                          in filtered_boxes, respectively
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for uclass in np.unique(box_classes):
            idxs = np.where(box_classes == uclass)

            # Filtered boxes, scores per box, and predictions for current class
            fboxs = filtered_boxes[idxs]
            b_scores = box_scores[idxs]
            b_classes = box_classes[idxs]

            idxs = b_scores.argsort()[::-1]

            # Corners of the bounding box
            x1, y1, x2, y2 = fboxs[:, 0], fboxs[:, 1], fboxs[:, 2], fboxs[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            keep = []

            while len(idxs) > 0:
                i = idxs[0]
                keep.append(i)  # Keep the index of the highest scoring box

                # Calculate overlap (Intersection over Union)
                xx1 = np.maximum(x1[i], x1[idxs[1:]])
                yy1 = np.maximum(y1[i], y1[idxs[1:]])
                xx2 = np.minimum(x2[i], x2[idxs[1:]])
                yy2 = np.minimum(y2[i], y2[idxs[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h

                iou = inter / (areas[i] + areas[idxs[1:]] - inter)

                inds = np.where(iou <= self.nms_t)[0]
                idxs = idxs[1:][inds]  # Update idxs to remove overlapped

            box_predictions.append(fboxs[keep])
            predicted_box_classes.append(b_classes[keep])
            predicted_box_scores.append(b_scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    def load_images(self, folder_path):
        """  Returns a list of images and their paths
            - folder_path: folder to inspect
        """
        image_paths = glob.glob(folder_path + "/*")


        return [cv2.imread(img) for img in image_paths], image_paths

    def preprocess_images(self, images):
        """ Preprocesses the images
            - images: images to preprocess
        """

        w = self.model.input.shape[1]
        h = self.model.input.shape[2]

        count = len(images)

        pimages = np.zeros((count, h, w, 3))
        image_shapes = []

        for i in range(count):
            img = cv2.resize(images[i], (w, h),
                             interpolation=cv2.INTER_CUBIC)
            pimages[i] = img / 255

            image_shapes.append(images[i].shape[0:-1])

        return (pimages, np.array(image_shapes))
