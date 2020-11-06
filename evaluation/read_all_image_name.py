# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import os

# read a folder, return the complete path
def get_files(path):
    ret = []
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(os.path.join(root,filespath)) 
    return ret

# read a folder, return the image name
def get_jpgs(path):
    ret = [] 
    for root, dirs, files in os.walk(path):  
        for filespath in files: 
            ret.append(filespath) 
    return ret

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

if __name__ == '__main__':
    fullname = get_files("C:\\Users\\ZHAO Yuzhi\\Desktop\\dataset\\COCO2014_val256_saliency")
    print("fullname saved")
    jpgname = get_jpgs("D:\\dataset\\COCO2014\\test2014")
    print("jpgname saved")
    text_save(fullname, "./COCO2014_val_256_saliency.txt")
    text_save(jpgname, "./COCO2014_test_name.txt")
    print("successfully saved")
    '''
    a = text_readlines("C:\\Users\\ZHAO Yuzhi\\Desktop\\code\\Colorization\\ILSVRC2012_train_name.txt")
    print(len(a))
    '''
