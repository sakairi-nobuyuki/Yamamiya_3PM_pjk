# coding: utf-8

import cv2
import numpy as np
from typing import List, Tuple


class MovingAverageTracker:
    img_ave: np.ndarray = np.array([0])
    old_img: np.ndarray = np.array([0])

    def __init__(self):
        pass

    def traking(self, input: np.ndarray) -> np.ndarray:

        gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        
        if self.old_img.size == 1:
            print("hoge")
            self.old_img = gray.copy().astype(float)
        print(gray.shape, self.old_img.shape, self.old_img.size)
        cv2.accumulateWeighted(gray, self.old_img, 0.5)
        
        diff_gray = cv2.absdiff(gray, cv2.convertScaleAbs(self.old_img))

        # thresh = cv2.threshold(diff_gray, 3, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.threshold(diff_gray, 100, 255, cv2.THRESH_BINARY)[1]
        self.thresh = thresh
        
        contours = cv2.findContours(thresh,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)[0]

        img_list = self.__create_cropped_list(input, contours)

        return img_list

    def __create_cropped_list(self, input: np.ndarray, contours: List[Tuple[int]]) -> np.ndarray:

        img_list = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 or h > 10:
                cv2.rectangle(input, (x, y), (x+w, y+h), (0, 0, 255), 2)
                # img_list.append(input[x: x+w, y: y+h])

        return img_list

