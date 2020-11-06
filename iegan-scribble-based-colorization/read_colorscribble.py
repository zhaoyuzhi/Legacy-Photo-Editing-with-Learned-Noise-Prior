import os
import numpy as np
import cv2

if __name__ == "__main__":
    
    img = cv2.imread('E:\\code\\Legacy Photo Editing\\data\\ILSVRC2012_val_256_colorscribble\\ILSVRC2012_val_00000005.JPEG')
    print(img.shape)
    for i in range(256):
        for j in range(256):
            if img[i, j, 0] > 0:
                print(img[i, j, 0], img[i, j, 1], img[i, j, 2])
    cv2.imshow('show', img)
    cv2.waitKey(0)
