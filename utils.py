"""
Created on Dec 29 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import numpy as np
import tensorflow as tf
import nibabel

# correlation calculation for keras metric and loss classes
def calculate_correlation(y_true, y_pred, sample_weight=None):
    assert len(y_true.shape)==5
    mean_ytrue = tf.reduce_mean(y_true, keepdims=True, axis=[1,2,3,4])
    mean_ypred = tf.reduce_mean(y_pred, keepdims=True, axis=[1,2,3,4])

    demean_ytrue = y_true - mean_ytrue
    demean_ypred = y_pred - mean_ypred

    if sample_weight is not None:
        sample_weight = tf.broadcast_weights(sample_weight, y_true)
        std_y = tf.sqrt(tf.reduce_sum(sample_weight * tf.square(demean_ytrue)) * tf.reduce_sum(
            sample_weight * tf.square(demean_ypred)))
        correlation = tf.reduce_sum(sample_weight * demean_ytrue * demean_ypred) / std_y
    else:
        std_y = tf.sqrt(tf.reduce_sum(tf.square(demean_ytrue)) * tf.reduce_sum(tf.square(demean_ypred)))
        correlation = tf.reduce_sum(demean_ytrue * demean_ypred) / std_y
    return tf.maximum(tf.minimum(correlation, 1.0), -1.0)

# correlation metric
class CorrelationMetric(tf.keras.metrics.Metric):
    def __init__(self, name="correlation", **kwargs):
        super(CorrelationMetric, self).__init__(name, **kwargs)
        self.correlation = self.add_weight(name='correlation', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        correlation = calculate_correlation(y_true, y_pred, sample_weight)
        self.correlation.assign(correlation)

    def result(self):
        return self.correlation


# correlation as loss function
class CorrelationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        correlation = calculate_correlation(y_true, y_pred, sample_weight)
        return 1.0 - correlation


# create input datasets as a sequence
class NiiSequence(tf.keras.utils.Sequence):
    def __init__(self, subIDs, rootpath, dataname, labelname, batch_size, shuffle=False):
        self.subIDs = subIDs
        self.batch_size = batch_size
        self.rootpath = rootpath
        self.dataname = dataname
        self.labelname = labelname
        self.shuffle = shuffle

    def __len__(self):
        return np.ceil(len(self.subIDs) / self.batch_size).astype(np.int64)

    def __getitem__(self, idx):
        if self.shuffle and idx == 0:
            shuffle_ids = np.arange(len(self.subIDs))
            np.random.shuffle(shuffle_ids)
            self.subIDs = np.array(self.subIDs)[shuffle_ids]
        subID_batch = self.subIDs[idx * self.batch_size:(idx + 1) * self.batch_size]
        data_batch = []
        label_batch = []
        for subID in subID_batch:
            data = nibabel.load(self.rootpath + subID + self.dataname).get_fdata()
            label = nibabel.load(self.rootpath + subID + self.labelname).get_fdata()
            label = np.expand_dims(label, axis=3)

            data_batch.append(data)
            label_batch.append(label)
        if self.batch_size>1:
            return np.stack(data_batch, axis=0), np.stack(label_batch, axis=0)
        else:
            return np.expand_dims(data, axis=0), np.expand_dims(label, axis=0)
