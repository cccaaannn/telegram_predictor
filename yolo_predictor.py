import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import cv2

class yolo_predictor():
    def __init__(self, model_path, names_path):
        self.__load_model(model_path)        
        self.__load_class_names(names_path)

    def __load_model(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def __load_class_names(self, names_path):
        self.names = {}
        with open(names_path, 'r') as data:
            for index, name in enumerate(data):
                self.names[index] = name.strip('\n')

    def __load_image(self, image_path, image_size):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(image, (image_size, image_size))
        image_data = image_data / 255.

        return image_data


    def predict(self, image_path, image_size=416, iou_threshold=0.45, score_threshold=0.25, max_output_size_per_class=50, max_total_size=50):
        """
        Predicts images with tensorflow converted darknet yolov4 model.

        Returns:
        A list of this tuple (class_name, class_index, confidence, (x, y, w, h))
        """

        # load image
        image_data = self.__load_image(image_path, image_size)

        # predict
        infer = self.model.signatures['serving_default']
        pred_bbox = infer(tf.constant(np.asarray([image_data]).astype(np.float32)))

        # cuting bbox and confidences
        # example shape for coco  pred_bbox=(1, 25, 84) boxes=(1, 25, 4) pred_conf=(1, 25, 80)  25 predictions form 80 classes before non max suppression 
        boxes = list(pred_bbox.values())[0][:, :, 0:4]
        pred_conf = list(pred_bbox.values())[0][:, :, 4:]

        # non max suppression
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=max_output_size_per_class,
            max_total_size=max_total_size,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        # convert tensors to numpy arrays
        boxes = boxes.numpy()
        scores = scores.numpy()
        classes = classes.numpy()
        num_boxes = valid_detections.numpy()

        predictions = []
        for i in range(num_boxes[0]):
            if(int(scores[0][i]) < 0 or int(scores[0][i]) > len(self.names)): 
                continue

            coor = boxes[0][i]
            x1 = coor[1]
            y1 = coor[0]
            x2 = coor[3]
            y2 = coor[2]

            cx = (x2 - (x2-x1) / 2) 
            cy = (y2 - (y2-y1) / 2) 

            w = (x2-x1)
            h = (y2-y1)

            predictions.append(
                (
                    self.names[classes[0][i]],  # class name
                    int(classes[0][i]),         # class index
                    scores[0][i],               # confidence
                    (cx,cy,w,h)                 # bbox
                )
            )

        return predictions



