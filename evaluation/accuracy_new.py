# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

import os
from indexes_new import *

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

def get_jpgs(path):
    # read a folder, return the image name
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

# New indexes accuracy for dataset
def Dset_Acuuracy(imglist, basepath):
    # Define the list saving the accuracy
    CNIlist = []
    CCIlist = []
    CCI_determinelist = []
    CNIratio = 0
    CCIratio = 0
    CCI_determineratio = 0

    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]
        imgpath = os.path.join(basepath, imgname)
        # Compute the traditional indexes
        CNI = SingleImageCNI(imgpath)
        CCI, CCI_determine = SingleImageCCI(imgpath)
        CNIlist.append(CNI)
        CCIlist.append(CCI)
        CCI_determinelist.append(CCI_determine)
        CNIratio = CNIratio + CNI
        CCIratio = CCIratio + CCI
        CCI_determineratio = CCI_determineratio + CCI_determine
        print('The %dth image: CNI: %f, CCI: %f, CCI_determine: %d' % (i, CNI, CCI, CCI_determine))
    CNIratio = CNIratio / len(imglist)
    CCIratio = CCIratio / len(imglist)
    CCI_determineratio = CCI_determineratio / len(imglist)

    return CNIlist, CCIlist, CCI_determinelist, CNIratio, CCIratio, CCI_determineratio
    
if __name__ == "__main__":
    
    # Read all names
    imglist = get_jpgs('C:\\Users\\yzzha\\Desktop\\dataset\\ILSVRC2012_val_256')

    # Define imgpath
    basepath = 'E:\\code\\Legacy Photo Editing\\data\\ILSVRC2012_val_256_CIC_colorized(final_step)'
    
    CNIlist, CCIlist, CCI_determinelist, CNIratio, CCIratio, CCI_determineratio = Dset_Acuuracy(imglist, basepath)

    print('The overall results: CNI: %f, CCI: %f, CCI_determine in [16, 20]: %f' % (CNIratio, CCIratio, CCI_determineratio))

    # Save the files
    base = ''
    text_save(CNIlist, "./statistic_files/%s_CNIlist.txt" % (base))
    text_save(CCIlist, "./statistic_files/%s_CCIlist.txt" % (base))
    text_save(CCI_determinelist, "./statistic_files/%s_CCI_determinelist.txt" % (base))
