# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:29:55 2018

@author: yzzhao2
"""

import numpy as np
import torch
from PIL import Image

# --------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------

# This is a look-up table generator for ImageNet2012
def mapping_scheme(filename):
    
    # Input: 'imagenet_2012_challenge_label_map_proto.txt'
    # Output: two lists look-up table representing the mapping scheme (target_class_string ---> target_class)
    # Example: target_class_string: n02481823
    #          target_class: 96
    
    content = text_readlines(filename)
    string_list = content[2:4000:4]
    scalar_list = content[1:4000:4]
    
    for i in range(1000):
        string_list[i] = string_list[i][24:33]
        scalar_list[i] = int(scalar_list[i][16:])
    
    return string_list, scalar_list

string_list, scalar_list = mapping_scheme('imagenet_2012_challenge_label_map_proto.txt')

# This is a mapping operation on validation set of ImageNet2012
def mapping_val(filename, string_list, scalar_list):
    
    # Input: 'imagenet_2012_validation_synset_labels.txt'
    # Output: The corresponding target_class list
    # Example: input_class_string: n02481823
    #          target_class: 96
    
    content = text_readlines(filename)
    val_scalar_list = []
    
    for i in range(50000):
        # The i-th component content[i]
        for index, value in enumerate(string_list):
            if value == content[i]:
                scalar = scalar_list[index]
                val_scalar_list.append(scalar)
    
    return val_scalar_list

val_string_list = text_readlines('imagenet_2012_validation_synset_labels.txt')
val_scalar_list = mapping_val('imagenet_2012_validation_synset_labels.txt', string_list, scalar_list)

for index, value in enumerate(scalar_list):
    if value == 490:
        scalar = string_list[index]
        print(scalar)
'''
# Saving
text_save(string_list, 'mapping_string.txt')
text_save(scalar_list, 'mapping_scalar.txt')
text_save(val_string_list, 'val_string.txt')
text_save(val_scalar_list, 'val_scalar.txt')
'''
