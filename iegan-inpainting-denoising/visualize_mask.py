import cv2
import random
import numpy as np

if __name__ == "__main__":
    
    img = cv2.imread('example.JPEG', flags = cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('./huahen/processed/5.png', flags = cv2.IMREAD_GRAYSCALE)

    # randomly crop
    rand_h = random.randint(0, max(0, mask.shape[0] - 256))
    rand_w = random.randint(0, max(0, mask.shape[1] - 256))
    mask = mask[rand_h:rand_h + 256, rand_w:rand_w + 256]
    
    img = (img.astype(np.float64) / 255) * (1 - mask.astype(np.float64) / 255)
    noise = np.random.normal(loc = 0.0, scale = 0.03, size = img.shape)
    masked_img = img + noise
    masked_img = np.clip(masked_img, 0, 1)
    masked_img = (masked_img * 255.0).astype(np.uint8)

    cv2.imshow('img', masked_img)
    cv2.waitKey(0)

    for i in range(1000):
        print(random.randint(1, 12))
    