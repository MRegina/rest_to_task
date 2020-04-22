"""
Created on Dec 29 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import numpy as np
import tensorflow as tf
import nibabel
import sys
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                            MaxPooling2D, Concatenate, UpSampling2D,
                            Conv3D, Conv3DTranspose, MaxPooling3D,
                            UpSampling3D, ZeroPadding3D, Dropout, 
                            SpatialDropout3D, BatchNormalization)
from tensorflow.keras import optimizers as opt
from tensorflow.keras import backend as K
                           
# correlation calculation for keras metric and loss classes
def correlation(y_true, y_pred, sample_weight=None):
   # assert len(y_true.shape)==5
    
    #### GET RID OF ZEROS ####
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    data_intersect = y_true * y_pred
    mask_intersect = tf.cast(data_intersect, dtype=tf.bool)

    y_true = tf.boolean_mask(y_true, mask_intersect)
    y_pred = tf.boolean_mask(y_pred, mask_intersect)

    mean_ytrue = tf.reduce_mean(y_true, keepdims=True)
    mean_ypred = tf.reduce_mean(y_pred, keepdims=True)
    	
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
        
    #tf.print("correlation:", correlation,output_stream=sys.stdout)
        
    #tf.print("correlation:", correlation)    
    return tf.maximum(tf.minimum(correlation, 1.0), -1.0)

def correlation_thresh(y_true, y_pred, thresh=2.58, sample_weight=None):
    
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    #### THRESHOLD DATA ####
    data_intersect = tf.cast(tf.math.greater(y_true,thresh),dtype=tf.float32) * y_pred
    mask_intersect = tf.cast(data_intersect, dtype=tf.bool)
    y_true = tf.boolean_mask(y_true, mask_intersect)
    y_pred = tf.boolean_mask(y_pred, mask_intersect)

    mean_ytrue = tf.reduce_mean(y_true, keepdims=True)
    mean_ypred = tf.reduce_mean(y_pred, keepdims=True)
  	
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
        
    #tf.print("correlation:", correlation,output_stream=sys.stdout)
        
    return tf.maximum(tf.minimum(correlation, 1.0), -1.0)

def correlation_gm(y_true, y_pred, sample_weight=None):

    sz = K.ndim(y_true)
    gm = nibabel.load('/data/Templates/Yeo2011_17Networks_2mm_LiberalMask_64.nii.gz').get_fdata()
    
    if K.eval(sz) == 5:
        gm = np.expand_dims(gm, axis=[0,-1])
    
    gm = tf.cast(gm,tf.bool)
    
    #### GM Mask ####
    y_true = tf.boolean_mask(y_true, gm)
    y_pred = tf.boolean_mask(y_pred, gm)

    mean_ytrue = tf.reduce_mean(y_true, keepdims=True)
    mean_ypred = tf.reduce_mean(y_pred, keepdims=True)
  
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

def mse_gm(y_true, y_pred):

    gm = nibabel.load('/data/Templates/Yeo2011_17Networks_2mm_LiberalMask_64.nii.gz').get_fdata()
    gm = np.expand_dims(gm, axis=[0,4])
    gm = tf.cast(gm,tf.bool)

    #### GM Mask ####
    y_true = tf.boolean_mask(y_true, gm)
    y_pred = tf.boolean_mask(y_pred, gm)

    loss = tf.square(y_true - y_pred)
      
    return loss
    
# correlation metric
class CorrelationMetric(tf.keras.metrics.Metric):
    def __init__(self, name="correlation", **kwargs):
        super(CorrelationMetric, self).__init__(name, **kwargs)
        self.correlation = self.add_weight(name='correlation', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        correlation = correlation(y_true, y_pred, sample_weight=None)
        self.correlation.assign(correlation)

    def result(self):
        return self.correlation

# correlation as loss function
class CorrelationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred, sample_weight=None):
        corr = correlation(y_true, y_pred, sample_weight=None)
        return 1.0 - corr
        
# create input datasets as a sequence
class NiiSequence(tf.keras.utils.Sequence):
    def __init__(self, subIDs, rootpath, dataname, labelname, labelnum, batch_size, thresh=None, shuffle=False):
        self.subIDs = subIDs
        self.batch_size = batch_size
        self.rootpath = rootpath
        self.dataname = dataname
        self.labelname = labelname
        self.labelnum = labelnum
        self.shuffle = shuffle
        self.thresh = thresh
        
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
            print(subID)
            data = nibabel.load(self.rootpath + 'TaskPredicted/Features/' + subID + self.dataname).get_fdata()
            if self.labelname is not None:
                label = nibabel.load(self.rootpath + subID + '/task/tfMRI_' + self.labelname + '/tfMRI_' + self.labelname + '_hp200_s4_level2vol.feat/cope' + self.labelnum + '.feat/stats/zstat1_64.nii.gz').get_fdata()
            
            if self.thresh == 'GM' or self.thresh == 'gm' or self.thresh == 'Gm':
                gm = nibabel.load('/data/Templates/Yeo2011_17Networks_2mm_LiberalMask_64.nii.gz').get_fdata()
                gm = (gm >= 1) * 1
                gm = gm.astype(np.int8)
                label = label * gm
                gm = np.tile(np.expand_dims(gm, axis=3),32)
                data = data * gm
            elif self.thresh is not None:
                label = K.greater(label,self.thresh)
                label = K.cast(label,"int64")
                
            if self.labelname is not None:
                label = np.expand_dims(label, axis=3)
                label_batch.append(label)
            
            data_batch.append(data)

        if self.batch_size>1:
            return np.stack(data_batch, axis=0), np.stack(label_batch, axis=0)
        else:
            if self.labelname is not None:
                return np.expand_dims(data, axis=0), np.expand_dims(label, axis=0)
            else:
                return np.expand_dims(data, axis=0)
        
# save predicted images in niftii format for later tests and visual checking
def save_prediction(predicted_batch, rootpath, outpath, template_subID, labelname, labelnum, batch_id=None, subIDs=None):
    template_img = nibabel.load('/media/Drobo_HCP/HCP_Data/Volume/CNN/100307_motor1_64.nii.gz')
    batch_size = predicted_batch.shape[0]
    for i in range(batch_size):
        new_img = nibabel.Nifti1Image(predicted_batch[i, :, :, :, :], template_img.affine)
        if subIDs is not None:
            filename = outpath + subIDs[i] + '_predicted_' + labelname + labelnum + '_UNET.nii.gz'
        elif batch_id is not None:
            filename = outpath + str(batch_id * batch_size + i) + '_predicted_' + labelname + labelnum + '_UNET.nii.gz'
        else:
            filename = outpath + str(i) + '_predicted_' + labelname + '_UNET.nii.gz'
        nibabel.save(new_img, filename)

# save predicted images in niftii format for later tests and visual checking
def save_prediction_batch(predicted_batch, rootpath, outpath, template_subID, labelname, labelnum, numtrain, batch_id=None, subIDs=None):
    os.makedirs(outpath,exist_ok=True)
    template_img = nibabel.load(rootpath + 'CNN/100307_motor1_64.nii.gz')
    batch_size = predicted_batch.shape[0]
    for i in range(batch_size):
        #new_img = nibabel.Nifti1Image(predicted_batch[i, :, :, :, :], template_img.affine, template_img.header)
        new_img = nibabel.Nifti1Image(predicted_batch[i, :, :, :, :], template_img.affine)
        if subIDs is not None:
            filename = outpath + subIDs[i] + '_predicted_' + str(numtrain) + '_UNET.nii.gz'
        elif batch_id is not None:
            filename = outpath + str(batch_id * batch_size + i) + '_predicted_' + str(numtrain) + '_UNET.nii.gz'
        else:
            filename = outpath + str(i) + '_predicted_' + str(numtrain) + '_UNET.nii.gz'
        nibabel.save(new_img, filename)
               
def load_nifti(test_ids, rootpath, labelname, labelnum, batch_id=None, subIDs=None, thresh=None):
    test_batch=[]
    for i in range(len(test_ids)):
        #print(test_ids[i])
        label = nibabel.load(rootpath + test_ids[i] + '/task/tfMRI_' + labelname + '/tfMRI_' + labelname + '_hp200_s4_level2vol.feat/cope' + labelnum + '.feat/stats/zstat1_64.nii.gz').get_fdata()  
        test_batch.append(label)
    if thresh is not None:
        test_batch = K.greater(test_batch,thresh)
        test_batch = K.cast(test_batch,"int64")  
    return np.array(test_batch) 
        
def act_pred_corr(predicted_batch, task_batch):
    #template_img =nibabel.load(rootpath + template_subID + '/task/tfMRI_' + labelname + '/tfMRI_' + labelname + '_hp200_s4_level2vol.feat/cope' + labelnum + '.feat/stats/tstat1.nii.gz').get_fdata()
    cc=np.zeros((predicted_batch.shape[0],task_batch.shape[0]))
    for i in range(predicted_batch.shape[0]):
        for j in range(task_batch.shape[0]):
	        tmp = correlation_gm(tf.cast(predicted_batch[i,:,:,:,0],tf.float32), tf.cast(task_batch[j,:,:,:],tf.float32), sample_weight=None)
	        tmp = tf.keras.backend.get_value(tmp)
	        cc[i,j] = tmp
    return cc
        
def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5,5,5),
                        deconvolution_kernel_size=(5,5,5),
                        pool_size=(2,2,2),
                        strides=(1,1,1),
                        mode='classification',
                        output_activation='tanh',
                        activation='relu',
                        init_lr=0.0001,
                        dropout=0.2,
                        dropout_type='spatial',
                        batchnorm=False):
    """
    Create a 3D Unet model
    Example
    -------
    unet_model = create_unet_model3D( (128,128,128,1), 1, 4)
    """
    layers = np.arange(layers)
    number_of_classification_labels = n_labels
    
    inputs = Input(shape=input_image_size)

    ## ENCODING PATH ##

    encoding_convolution_layers = []
    pool = None
    conv_tot=0
    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2**(layers[i])
        conv_tot += 1
        if i == 0:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation=activation,
                            padding='same',name='conv3d_' + str(conv_tot))(inputs)
        else:
            conv = Conv3D(filters=number_of_filters, 
                            kernel_size=convolution_kernel_size,
                            activation=activation,
                            padding='same',name='conv3d_' + str(conv_tot))(pool)

        if dropout is not None: 
            if dropout_type is 'spatial':
                conv = SpatialDropout3D(dropout)(conv)
            else:
                conv = Dropout(dropout)(conv)                
        if batchnorm is True: 
            conv = BatchNormalization(axis=4)(conv)    
        
        conv_tot += 1
        encoding_convolution_layers.append(Conv3D(filters=number_of_filters, 
                                                        kernel_size=convolution_kernel_size,
                                                        activation=activation,
                                                        padding='same',name='conv3d_' + str(conv_tot))(conv))
		
        if i < len(layers)-1:
            pool = MaxPooling3D(pool_size=pool_size,name='pool_' + str(i))(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers)-1]
    conv_tot += 1
    print(conv_tot)
    for j in range(1,len(layers)):
        if j < len(layers)-1:
            decon_kernel_size = deconvolution_kernel_size
        else:
            decon_kernel_size = deconvolution_kernel_size
            
        number_of_filters = lowest_resolution * 2**(len(layers)-layers[j]-1)
        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=decon_kernel_size,
                                     padding='same', name='trans_' + str(j))(outputs)
        tmp_deconv = UpSampling3D(size=pool_size,name='upsamp_' + str(j))(tmp_deconv)
        outputs = Concatenate(axis=4,name='concat_' + str(j))([tmp_deconv, encoding_convolution_layers[len(layers)-j-1]])
		
        conv_tot += 1
        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation=activation, padding='same',name='conv3d_' + str(conv_tot))(outputs)
                        
        if dropout is not None: 
            if dropout_type is 'spatial':
                outputs = SpatialDropout3D(dropout)(outputs)
            else:
                outputs = Dropout(dropout)(outputs) 
        if batchnorm is True: 
            outputs = BatchNormalization(axis=4)(outputs)   
        
        conv_tot += 1
        outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, 
                        activation=activation, padding='same',name='conv3d_' + str(conv_tot))(outputs)

        if dropout is not None: 
            if dropout_type is 'spatial':
                outputs = SpatialDropout3D(dropout)(outputs)
            else:
                outputs = Dropout(dropout)(outputs) 
        if batchnorm is True: 
            outputs = BatchNormalization(axis=4)(outputs)   	
            	    
    if mode == 'classification':
        if number_of_classification_labels == 1:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='sigmoid')(outputs)
        else:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1,1,1), 
                            activation='softmax')(outputs)

        unet_model = Model(inputs=inputs, outputs=outputs)

        if number_of_classification_labels == 1:
            unet_model.compile(loss=jaccard_distance,
                                optimizer=opt.Adam(lr=init_lr), metrics=['binary_accuracy','binary_crossentropy',dice_coefficient_bin])
        else:
            unet_model.compile(loss='categorical_crossentropy', 
                                optimizer=opt.Adam(lr=init_lr), metrics=['categorical_accuracy', 'categorical_crossentropy'])
    elif mode =='regression':
        conv_tot += 1
        outputs = Conv3D(filters=1, kernel_size=(1,1,1), 
                        activation=output_activation,name='conv3d_' + str(conv_tot))(outputs)
        
        unet_model = Model(inputs=inputs, outputs=outputs)
        unet_model.compile(loss='mse', optimizer=opt.Adam(lr=init_lr), metrics=[correlation, correlation_gm, mse_gm])        
    
    else:
        raise ValueError('mode must be either `classification` or `regression`')

    return unet_model
