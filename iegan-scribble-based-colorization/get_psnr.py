import cv2
import numpy as np
import math

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

PATH = "C:/Users/a/Desktop/legacy_image_editing/result1/"

psnr_sum = 0

for i in range(3):

    img = cv2.imread(PATH + "image%d.jpg"%i)
    #print(img)
    fake = cv2.imread(PATH + "result_%d.jpg"%i)
    psnr_i = psnr(fake, img)
    psnr_sum += psnr_i
    print(psnr_i)

print("psnr_mean", psnr_sum/3)