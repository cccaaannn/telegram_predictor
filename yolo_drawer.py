import numpy as np
import random
import cv2
import os

class yolo_drawer():
    def __init__(self, random_seed_start_point=42):
        self.random_seed_start_point = random_seed_start_point

    def __convert_points(self, image_width, image_height, points):
        """
        Converts yolo points (x,y,w,h) to x1,y1,x2,y2
        """
        x1 = int(image_width * (points[0] - points[2] / 2))  # top left xd
        y1 = int(image_height * (points[1] - points[3] / 2)) # top left y
        x2 = int(image_width * (points[0] + points[2] / 2))  # bottom right x
        y2 = int(image_height * (points[1] + points[3] / 2)) # bottom right y
        return x1, y1, x2, y2


    def draw(self, predictions, image_path, show=True, save_folder_path=None, resize=(1280, 720), saved_file_suffix="_predicted"):
        """
        Draws bounding boxes of predicted images.

        Predictions has to be in this format (class_name, class_index, confidence, (x, y, w, h))

        Returns:
        opencv image object
        """

        # read image
        image = cv2.imread(image_path)
        if(resize):
            image = cv2.resize(image, tuple(resize))

        # get image dimensions
        image_height = np.size(image, 0)
        image_width = np.size(image, 1)

        # draw the rectangles
        for prediction in predictions:

            confidence = float("{0:.2f}".format(float(prediction[2]*100)))
            name = prediction[0]

            # create color with seed for consitency
            random.seed(prediction[1] + self.random_seed_start_point)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # convert points
            x1, y1, x2, y2 = self.__convert_points(image_width, image_height, prediction[3])

            # draw
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(image, "{0} %{1}".format(name, confidence), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)

        # save image
        save_path = None
        if(save_folder_path):
            # make the dir if not exists
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            # set up save location
            image_name, image_extension = os.path.splitext(os.path.basename(image_path))
            new_file_name = "{0}{1}{2}".format(image_name, saved_file_suffix, image_extension)
            save_path = os.path.join(save_folder_path, new_file_name)

            # save
            cv2.imwrite(save_path, image)

        # show image
        if(show):
            cv2.namedWindow(image_path)
            cv2.moveWindow(image_path, 50,50)
            cv2.imshow(image_path, image)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return image, save_path
