import cv2
import numpy as np

class houghModel():

    @classmethod
    def get_error(hough_pred, y_true):
        pass

    def __init__(self, canny_threshold1 = 50, canny_threshold2 = 50,
                hough_rho = 1, hough_theta = 1, hough_threshold = 150):
        
        self.canny_thr1 = canny_threshold1
        self.canny_thr2 = canny_threshold2
        self.hough_r = hough_rho
        self.hough_t = hough_theta
        self.hough_thr = hough_threshold
        self.hough_arr = []
        self.img_arr = []
 
    def make_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def apply_canny(self, img_gray):
        return cv2.Canny(img_gray, self.canny_thr1,
                                self.canny_thr2)
    
    def find_hough_stimation(self, img_edges):
        self.hough_arr.append(cv2.HoughLines(img_edges, self.hough_r, 
                                            self.hough_t, self.hough_thr))

    def fit(self, img):
        self.img_arr.append(img)

        #Img preprocesing
        self.img_gray = self.make_gray(img)
        self.img_canny =  self.apply_canny(self.img_gray)
        
        #Found lines with Hough
        self.hough_arr.append(self.find_hough_stimation(self.img_canny))


    def predict(self):

        output_img = []
        output_lines = []
        for line_pred, or_img in zip(self.img_arr, self.hough_arr):
            aux_arr = []
            for line in line_pred:
                rho,theta = line[0]
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
