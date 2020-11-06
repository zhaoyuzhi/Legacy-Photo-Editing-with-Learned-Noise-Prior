import os
import numpy as np
import cv2

# color scribble
def color_scribble(img, color_point = 30, color_width = 5):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    scribble = np.zeros((height, width, channels), np.uint8)

    times = np.random.randint(color_point)
    times = 30
    print(times)
    for i in range(times):
        # random selection
        rand_h = np.random.randint(height)
        rand_w = np.random.randint(width)
        # define min and max
        min_h = rand_h - (color_width - 1) // 2
        max_h = rand_h + (color_width - 1) // 2
        min_w = rand_w - (color_width - 1) // 2
        max_w = rand_w + (color_width - 1) // 2
        min_h = max(min_h, 0)
        min_w = max(min_w, 0)
        max_h = min(max_h, height)
        max_w = min(max_w, width)
        # attach color points
        scribble[min_h:max_h, min_w:max_w, :] = img[rand_h, rand_w, :]

    return scribble

def blurish(img, color_blur_width = 11):
    img = cv2.GaussianBlur(img, (color_blur_width, color_blur_width), 0)
    return img
    
def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

# multi-layer folder
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
if __name__ == "__main__":
    
    imglist = get_files('C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256')
    namelist = get_jpgs('C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256')

    for i in range(len(namelist)):
        img = cv2.imread(imglist[i])
        scribble = color_scribble(img)
        #scribble = blurish(scribble)

        savefolder = 'E:\\code\\Legacy Photo Editing\\data\\ILSVRC2012_val_256_colorscribble'
        check_path(savefolder)
        savepath = os.path.join(savefolder, namelist[i].split('.')[0] + '.png')
        print(i, savepath)
        
        '''
        show = np.concatenate((img, scribble), axis = 1)
        cv2.imshow('show', show)
        cv2.waitKey(1000)
        '''
        
        cv2.imwrite(savepath, scribble)
