import os
import cv2
import utils

imglist = utils.get_jpgs('./huahen/ori')
utils.check_path('./huahen/processed')

for i in range(len(imglist)):
    imgpath = os.path.join('./huahen/ori', imglist[i])
    img = cv2.imread(imgpath, flags = cv2.IMREAD_GRAYSCALE)
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print(i, thresh1.shape)
    if thresh1.shape[0] < 256 and thresh1.shape[0] <= thresh1.shape[1]:
        thresh1 = cv2.resize(thresh1, (int(thresh1.shape[1] / thresh1.shape[0] * 256), 256))
        print(i, 'processed', thresh1.shape)
    if thresh1.shape[1] < 256 and thresh1.shape[1] <= thresh1.shape[0]:
        thresh1 = cv2.resize(thresh1, (256, int(thresh1.shape[0] / thresh1.shape[1] * 256)))
        print(i, 'processed', thresh1.shape)
    savepath = os.path.join('./huahen/processed', imglist[i].split('.')[0] + '.png')
    cv2.imwrite(savepath, thresh1)


'''
img = cv2.imread('C:\\Users\\yzzha\\Desktop\\huahen\\1.jpeg')
print(img.shape)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('test', thresh1)
cv2.waitKey(0)

cv2.imwrite('C:\\Users\\yzzha\\Desktop\\huahen\\processed.png', thresh1)
'''
