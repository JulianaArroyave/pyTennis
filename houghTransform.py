import cv2
import numpy as np
from math import isclose

class houghModel():

    @classmethod
    def get_error(hough_pred, y_true):
        dist_min = 1000
        position = []

        x1, y1, x2, y2 = tuple(y_true)

        slope_r = (y1 - y2)/(x1 -x2)
        y = np.sqrt((y1-y2)**2)/2
        x = np.sqrt((x1 - x2)**2)/2

        for i in hough_pred[i]:
            
            x1_p = hough_pred[i][0]
            y1_p = hough_pred[i][1]
            x2_p = hough_pred[i][2]
            y2_p = hough_pred[i][3]

            x_p = np.sqrt((x2_p - x1_p)**2)/2
            y_p = np.sqrt((y2_p - y1_p)**2)/2

            slope_p = (y1_p - y1_p)/(x1_p - x2_p)


            dist = np.sqrt((x-x_p)*2 + (y-y_p)*2)

            if dist < dist_min and (isclose(slope_r, slope_p, abs_tol=10**-5)):
                dist_min = dist
                position.append(i)

    def __init__(self, canny_threshold1 = 20, canny_threshold2 = 20,
                hough_rho = 1, hough_theta = 1, hough_threshold = 150):
        
        self.canny_thr1 = canny_threshold1
        self.canny_thr2 = canny_threshold2
        self.hough_r = hough_rho
        self.hough_t = hough_theta
        self.hough_thr = hough_threshold
        self.img_arr = []
 
    def make_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_canny(self, img_gray):
        return cv2.Canny(img_gray, self.canny_thr1,
                                self.canny_thr2)
    
    def find_hough_stimation(self, img_edges):
        self.hough_arr = cv2.HoughLines(img_edges, self.hough_r, 
                                            self.hough_t, self.hough_thr)
        return  self.hough_arr

    def fit(self, img):
        self.img_arr.append(img)

        #Img preprocesing
        self.img_gray = self.make_gray(img)
        self.img_canny =  self.apply_canny(self.img_gray)
        
        #Found lines with Hough
        result = self.find_hough_stimation(self.img_canny)
        print(result)

    def predict(self):

        output_img = []
        output_lines = []
        for or_img, line_pred in zip(self.img_arr, self.hough_arr):
            aux_arr = []
            
            for line in line_pred:
                rho,theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                aux_arr.append([x1, y1, x2, y2])

                cv2.line(or_img,(x1,y1),(x2,y2),(0,0,255),2)

            output_img.append(or_img)
            output_lines.append(aux_arr)

        return output_img, output_lines

