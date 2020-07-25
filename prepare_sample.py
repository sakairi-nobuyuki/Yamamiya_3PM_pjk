import os
import glob
import shutil
import cv2
import numpy as np
import pprint


class DetectPlum:
    def __init__(self):
        #super().__init__()
        self.raw_sample_path = 'validation_g_img_search'
        self.img_path_list = glob.glob ('{}/*.jpg'.format (self.raw_sample_path))
        self.img_list = []
        #print (self.img_path_list)
        #self.load_img ()
        self.dest_dir_path  = 'validation_g_img_search'

        #if os.path.exists (self.dest_dir_path):
        #    shutil.rmtree (self.dest_dir_path)
        #os.makedirs (self.dest_dir_path)

        if os.path.exists (self.dest_dir_path) == False:
            os.makedirs (self.dest_dir_path)
        
    def shift_jpg2png (self):
        for img_path in self.img_path_list:
            ext_name = os.path.basename (img_path).split ('.') [1]
            print ("file name: {}, ext: {}".format (os.path.basename (img_path), ext_name))

            img = cv2.imread (img_path)

            new_img_path = self.rename_bbox_file (img_path)

            print ("changed to png: ", new_img_path)
            #if os.path.basename (img_path).split ('.') [1] == 'jpg':

            cv2.imwrite (new_img_path, img)

    def load_img (self):
        for img_path in self.img_path_list:
            img = cv2.imread (img_path)
            self.img_list.append (img)
            print ('load file: {}\n  w, h = {}, {}'.format (os.path.basename (img_path), img.shape[0], img.shape[1]))

    def find_plum_in_raw_img (self):
        #for img in self.img_list:
        for img_path in self.img_path_list:
            img = cv2.imread (img_path)
            print ('load file: {}\n  w, h = {}, {}'.format (os.path.basename (img_path), img.shape[0], img.shape[1]))

            img_gray   = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold (img_gray, 0, 255, cv2.THRESH_OTSU)

            #img_HSV    = cv2.cvtColor (img, cv2.COLOR_BGR2HSV)
            #img_H, img_S, img_V = cv2.split (img_HSV)
            #_, img_bin = cv2.threshold (img_H, 0, 255, cv2.THRESH_OTSU)

            contours, hierarchy = cv2.findContours (img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            img_contour = img.copy ()

            for contour in contours:
                if len (contour) == 0:  continue
                if cv2.contourArea (contour) < 100:  continue
                x, y, w, h = cv2.boundingRect (contour)
                cv2.rectangle (img_contour, (x, y), (x + w, y + h), (0, 0, 0), 10)
            print ("  new file name", self.rename_bbox_file (img_path))
            
            cv2.imwrite (self.rename_bbox_file (img_path), img_contour)
    
    def rename_bbox_file (self, img_path):
        return os.path.join (self.dest_dir_path, os.path.basename (img_path).replace ('jpg', 'png'))


if __name__ == '__main__':
    plum_detect = DetectPlum ()
    plum_detect.shift_jpg2png ()
    #    plum_detect.find_plum_in_raw_img ()



