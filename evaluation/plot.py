# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:36:00 2018

@author: yzzhao2
"""

from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# read the txt and return the float list
def text_plot(base = 'noSal', index = 'psnr'):
    # Define txt name
    txtname = base + '_' + index + 'list.txt'
    fullname = './statistic_files/' + txtname    
    # Read all names and change the dtype as float
    imglist = text_readlines(fullname)
    datalist = []
    for i in range(len(imglist)):
        datalist.append(float(imglist[i]))
    return datalist

# PSNR figure
def PSNR_kdeplot():
    # ----------------------------------------------------
    #    Plot the KDE approximation figure of one index
    # ----------------------------------------------------
    # Return the list containing all the data
    GAN6_BN_new_psnr_list = text_plot(base = 'GAN6_BN_new', index = 'psnr')
    noVGG16_psnr_list = text_plot(base = 'noVGG16', index = 'psnr')
    noSal_psnr_list = text_plot(base = 'noSal', index = 'psnr')
    noGAN_psnr_list = text_plot(base = 'noGAN', index = 'psnr')
    GT_gray_psnr_list = text_plot(base = 'GT_gray', index = 'psnr')
    dic = {"Ours (full)": GAN6_BN_new_psnr_list, "Ours (w/o global feature)": noVGG16_psnr_list,
        "Ours (w/o saliency map)": noSal_psnr_list, "Ours (w/o GAN)": noGAN_psnr_list,
        "Gray": GT_gray_psnr_list}
    frame = DataFrame(dic)
    sns.set(style = 'whitegrid')

    # Draw the histogram of PSNR
    # shade: shade below the histogram
    # kernel: KDE kernel type
    # bw: bandwidth of kernel
    # linewidth: width of line
    sns.kdeplot(frame['Ours (full)'], shade = False, kernel = 'gau', bw = 0.5, linewidth = 2)
    sns.kdeplot(frame['Ours (w/o global feature)'], shade = False, kernel = 'gau', bw = 0.5, linewidth = 2)
    sns.kdeplot(frame['Ours (w/o saliency map)'], shade = False, kernel = 'gau', bw = 0.5, linewidth = 2)
    sns.kdeplot(frame['Ours (w/o GAN)'], shade = False, kernel = 'gau', bw = 0.5, linewidth = 2)
    sns.kdeplot(frame['Gray'], shade = False, kernel = 'gau', bw = 0.5, linewidth = 2)

    # Set the style of figure
    plt.xlim([10, 40])
    xticks = np.linspace(10, 40, 7)
    plt.xticks(xticks)
    plt.xlabel('PSNR', weight = 'normal', size = 12)
    plt.ylabel('Frequency', weight = 'normal', size = 12)
    plt.show()

# PSNR figure
def PSNR_violinplot():
    # ----------------------------------------------------
    #    Plot the KDE approximation figure of one index
    # ----------------------------------------------------
    # Return the list containing all the data
    GAN6_BN_new_psnr_list = text_plot(base = 'GAN6_BN_new', index = 'psnr')
    noVGG16_psnr_list = text_plot(base = 'noVGG16', index = 'psnr')
    noSal_psnr_list = text_plot(base = 'noSal', index = 'psnr')
    noGAN_psnr_list = text_plot(base = 'noGAN', index = 'psnr')
    GT_gray_psnr_list = text_plot(base = 'GT_gray', index = 'psnr')
    dic = {"Ours (full)": GAN6_BN_new_psnr_list, "Ours (w/o global feature)": noVGG16_psnr_list,
        "Ours (w/o saliency map)": noSal_psnr_list, "Ours (w/o GAN)": noGAN_psnr_list,
        "Gray": GT_gray_psnr_list}
    frame = DataFrame(dic)
    sns.set(style = 'whitegrid')

    # Draw the histogram of PSNR
    # shade: shade below the histogram
    # kernel: KDE kernel type
    # bw: bandwidth of kernel
    # linewidth: width of line
    sns.violinplot(data = frame)#, orient = "v", saturation = 1, width = 0.3, fliersize = 0.8, linewidth = 2, whis = 1)

    # Set the style of figure
    plt.ylim([10, 40])
    yticks = np.linspace(10, 40, 7)
    plt.yticks(yticks)
    plt.xlabel('Methods', weight = 'normal', size = 12)
    plt.ylabel('PSNR', weight = 'normal', size = 12)
    plt.show()

# CCI figure
def CCI_boxplot():
    # ----------------------------------------------------
    #           Plot the box figure of one index
    # ----------------------------------------------------
    # Return the list containing all the data
    noSal_CCI_list = text_plot(base = 'noSal', index = 'CCI')
    Larsson_CCI_list = text_plot(base = 'Larsson', index = 'CCI')
    Isola_CCI_list = text_plot(base = 'Isola', index = 'CCI')
    Zhang_CCI_list = text_plot(base = 'Zhang', index = 'CCI')
    Iizuka_CCI_list = text_plot(base = 'Iizuka', index = 'CCI')
    dic = {"Ours (full)": noSal_CCI_list, "Larsson et al.": Larsson_CCI_list, "Isola et al.": Isola_CCI_list, "Zhang et al.": Zhang_CCI_list, "Iizuka et al.": Iizuka_CCI_list}
    frame = DataFrame(dic)
    sns.set(style = 'whitegrid')
    
    # Draw the line and point for CCI figure
    x1 = np.linspace(-1, -0.17, 2)
    x2 = np.linspace(0.17, 0.82, 2)
    x3 = np.linspace(1.17, 3.83, 2)
    x4 = np.linspace(4.17, 5, 2)
    y = np.array([16, 16])
    plt.plot(x1, y, color = 'lightgray', linestyle = '--', linewidth = 1)
    plt.plot(x2, y, color = 'lightgray', linestyle = '--', linewidth = 1)
    plt.plot(x3, y, color = 'lightgray', linestyle = '--', linewidth = 1)
    plt.plot(x4, y, color = 'lightgray', linestyle = '--', linewidth = 1)
    plt.text(-0.812, 14, '16', fontsize = 11)
    
    # Draw the box figure
    # orient: direction of the boxplot
    # saturation: color saturation
    # width: width of box
    # filtersize: size of outliers
    # linewidth: width of line
    # whis: filter the value of outliers
    sns.boxplot(data = frame, orient = "v", saturation = 1, width = 0.3, fliersize = 0.8, linewidth = 2, whis = 1)
    
    # Set the style of figure
    plt.ylim([0, 70])
    yticks = np.linspace(0, 70, 8)
    plt.yticks(yticks)
    plt.xlabel('Methods', weight = 'normal', size = 12)
    plt.ylabel('CCI', weight = 'normal', size = 12)
    plt.show()

if __name__ == "__main__":
    #PSNR_kdeplot()
    #PSNR_violinplot()
    CCI_boxplot()