# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import nibabel
import matplotlib.pyplot as plt

data_path = 'C:\\phd\\connectome_conv_net_tf2\\CNN\\'
subIDs = [100307, 101915, 103414, 103818, 106319]


def get_histogram(img_data):
    hist, bins = np.histogram(img_data, bins=np.arange(img_data.min(), img_data.max(), .1))
    stats = [np.median(img_data), np.percentile(img_data, 25), np.percentile(img_data, 75), np.percentile(img_data, 5),
             np.percentile(img_data, 95)]
    return hist, bins[:-1] + 0.05, stats


for subID in subIDs:
    rest_img = nibabel.load(data_path + str(subID) + '_DR2_nosmoothing.nii.gz')
    task_img = nibabel.load(data_path + str(subID) + '_motor1.nii.gz')

    rest_hist, rest_bins, rest_stats = get_histogram(rest_img.get_fdata())

    task_hist, task_bins, task_stats = get_histogram(task_img.get_fdata())

    plt.figure(figsize=[20, 16])
    plt.subplot(521)
    plt.plot(task_bins, task_hist)
    plt.title("50prc %.3f,25prc %.3f, 75prc %.3f, 5prc %.3f, 95prc %.3f" % tuple(task_stats))
    plt.subplot(522)
    plt.plot(rest_bins, rest_hist)
    plt.title("50prc %.3f,25prc %.3f, 75prc %.5f, 5prc %.3f, 95prc %.3f" % tuple(rest_stats))

    for i in range(8):
        hist, bins, stats = get_histogram(rest_img.get_fdata()[:, :, :, i * 4])
        plt.subplot(5, 2, 3 + i)
        plt.plot(bins, hist)
        plt.title("50prc %.3f,25prc %.3f, 75prc %.3f, 5prc %.3f, 95prc %.3f" % tuple(stats))

    plt.savefig('%s_histograms.png' % subID)

    plt.figure(figsize=[20, 20])
    for i in range(4):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rest_img.get_fdata()[:, :, 15 + i * 20, 10])
        plt.subplot(4, 4, 4 + i + 1)
        plt.imshow(rest_img.get_fdata()[:, :, 15 + i * 20, 20])
        plt.subplot(4, 4, 8 + i + 1)
        plt.imshow(rest_img.get_fdata()[:, :, 15 + i * 20, 30])

        plt.subplot(4, 4, 12 + i + 1)
        plt.imshow(task_img.get_fdata()[:, :, 15 + i * 20])

    plt.savefig('%s_slices.png' % subID)

    plt.figure(figsize=[20, 20])
    for i in range(4):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rest_img.get_fdata()[15 + i * 20, :, :, 10])
        plt.subplot(4, 4, 4 + i + 1)
        plt.imshow(rest_img.get_fdata()[15 + i * 20, :, :, 20])
        plt.subplot(4, 4, 8 + i + 1)
        plt.imshow(rest_img.get_fdata()[15 + i * 20, :, :, 30])

        plt.subplot(4, 4, 12 + i + 1)
        plt.imshow(task_img.get_fdata()[15 + i * 20, :, :, ])

    plt.savefig('%s_slices2.png' % subID)

    plt.figure(figsize=[20, 20])
    for i in range(4):
        plt.subplot(4, 4, i + 1)
        plt.imshow(rest_img.get_fdata()[:, 15 + i * 20, :, 10])
        plt.subplot(4, 4, 4 + i + 1)
        plt.imshow(rest_img.get_fdata()[:, 15 + i * 20, :, 20])
        plt.subplot(4, 4, 8 + i + 1)
        plt.imshow(rest_img.get_fdata()[:, 15 + i * 20, :, 30])

        plt.subplot(4, 4, 12 + i + 1)
        plt.imshow(task_img.get_fdata()[:, 15 + i * 20, :, ])

    plt.savefig('%s_slices3.png' % subID)
