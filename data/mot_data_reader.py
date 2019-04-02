import os
import pandas as pd
import cv2
import numpy as np


class MOTDataReader:
    def __init__(self, image_folder, detection_file_name, min_confidence = None):
        self.image_folder = image_folder
        self.detection_file_name = detection_file_name
        self.detection = pd.read_csv(self.detection_file_name, header=None)
        if min_confidence is not None:
            self.detection = self.detection[self.detection[6] > min_confidence]
        self.detection_group = self.detection.groupby(0)
        self.detection_group_keys = list(self.detection_group.indices.keys())
    def __len__(self):
        return len(self.detection_group_keys)

    def get_detection_by_index(self, index):
        return self.detection_group.get_group(index).values

    def get_image_by_index(self, index):
        print((os.path.join(self.image_folder, f'{index}.jpg')))
        return cv2.imread(os.path.join(self.image_folder, f'{index}.jpg'))
    def __getitem__(self, item):
        index=self.detection_group_keys[item]
        print("__getitem__",item)
        return (self.get_image_by_index(index),
                self.get_detection_by_index(index))


class DataTransform():
    @staticmethod
    def transform(image, detection, size, mean):
        '''
        transform image and detection to the sst input format
        :param image:
        :param detection:
        :param size:
        :param mean:
        :return:
        '''
        h, w, c = image.shape
        image.astype(np.float32)
        detection[[4, 5]] += detection[2, 3]
        image = cv2.resize(image, size)
        image -= mean
        new_h, new_w, new_c = image.shape

