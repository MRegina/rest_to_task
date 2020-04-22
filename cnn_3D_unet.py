"""
Created on Dec 28 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import tensorflow as tf
from datetime import datetime
import os
import numpy as np
import random
from utils import NiiSequence, create_unet_model3D, CorrelationMetric, CorrelationLoss, save_prediction, save_prediction_batch, correlation, correlation_thresh, load_nifti, act_pred_corr,correlation
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pickle

DATA_PATH = '/media/Drobo_HCP/HCP_Data/Volume/'
OUT_PATH = '/media/Drobo_HCP/HCP_Data/Volume/CNN/Predictions/'

num_val=50;
num_test=50;

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_train_alltasks.txt') as f:
    SUBIDS = f.read().split('\n')
SUBIDS.pop()

with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_val_alltasks.txt') as f:
    VALID_SUBIDS = f.read().split('\n')
VALID_SUBIDS.pop()
    
with open('/media/Drobo_HCP/HCP_Data/Volume/CNN/list_test_alltasks.txt') as f:
    TEST_SUBIDS = f.read().split('\n')
TEST_SUBIDS.pop()

#print(TRAIN_SUBIDS)
#print(VALID_SUBIDS)
#print(TEST_SUBIDS)

REST_FILE_NAME = '_DR2_4mm_64.nii.gz'
TASK_FILE_NAME = 'RELATIONAL'
TASK_FILE_NUM = '1'
# output paths

BATCH_SIZE = 1
#INPUT_SHAPE = (96, 112, 96, 32)
#OUTPUT_SIZE = (96, 112, 96, 1)
INPUT_SHAPE = (64, 64, 64, 32)
OUTPUT_SIZE = (64, 64, 64, 1)

numfilters=[96]
kersize=[3]
numlayers=[2]

num_trains=[10,20,50,100,200,300];
#num_trains=[200];
cc_mean_test_1D = np.zeros((24))
cc_mean_val_1D = np.zeros((24))


EPOCH=300
DROPOUT_RATE=None  # 0.1
batchnorm=False
 
count=-1      
for layer in numlayers:
    for ker in kersize:
        for filt in numfilters:
            for trainnum in num_trains:
                count += 1
            
                # hyperparameters
                layers=layer
                filt_num=filt
                kernel_size=ker
                
                print(trainnum, filt_num, kernel_size, layers)
                random.seed(6)
                TRAIN_SUBIDS = random.sample(SUBIDS, k=trainnum)
                LOGDIR = os.path.join("/media/Drobo_HCP/HCP_Data/Volume/logs/unet/param_test/", TASK_FILE_NAME + TASK_FILE_NUM, datetime.now().strftime("%Y%m%d"), datetime.now().strftime("%H%M%S") + '_layers' + str(layers) + '_nfilt' + str(filt_num) + '_kersz' + str(kernel_size) + '_ntrain' + str(trainnum) + '_do' + str(DROPOUT_RATE) + '_bn' + str(batchnorm))
                os.makedirs(os.path.join(LOGDIR, "checkpoints"), exist_ok=False)
                CHECKPOINT_PATH = os.path.join(LOGDIR, "checkpoints", "cp-{epoch:04d}.ckpt")
                
                # tf.config.experimental_run_functions_eagerly(True)
                
                # create datasets
                train_dataset = NiiSequence(TRAIN_SUBIDS, shuffle=True, rootpath=DATA_PATH, dataname=REST_FILE_NAME,
                                            labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM, batch_size=BATCH_SIZE, thresh=None)
                valid_dataset = NiiSequence(VALID_SUBIDS, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                                            batch_size=BATCH_SIZE, thresh=None)
                test_dataset = NiiSequence(TEST_SUBIDS, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM,
                                           batch_size=1, thresh=None)
                    
                # create unet model
                model = create_unet_model3D(input_image_size=INPUT_SHAPE, n_labels=32, layers=layers,
                                            mode='regression',output_activation='linear',strides=(1,1,1),
                                            pool_size=(2,2,2),lowest_resolution=filt_num, init_lr=0.0001,
                                            convolution_kernel_size=(kernel_size,kernel_size,kernel_size),
                                            deconvolution_kernel_size=(kernel_size,kernel_size,kernel_size),
                                            dropout=DROPOUT_RATE,batchnorm=batchnorm,dropout_type='spatial',
                                            activation='relu')
                
                # create callbacks
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR,profile_batch=0)
                #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, period=5)
                cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_correlation_gm',mode='max',save_best_only=True, save_weights_only=True)
                stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_correlation_gm', min_delta=0, patience=20, verbose=0, mode='max', baseline=None, restore_best_weights=True)
     
                model.summary()
                
                # save model architecture in json
                model_json = model.to_json()
                with open(LOGDIR + "/model.json", "w") as json_file:
                    json_file.write(model_json)
                
                # training loop
                model.fit(train_dataset, epochs=EPOCH, validation_data=valid_dataset, callbacks=[tensorboard_callback, cp_callback, stop_callback])
                
                # evaluate on the test and val sets
                test_loss = model.evaluate(test_dataset)
                #cc_mean_test[layernum][kernum][filtnum] = test_loss[2]
                cc_mean_test_1D[count] = test_loss[2]
                
                val_loss = model.evaluate(valid_dataset)
                #cc_mean_val[layernum][kernum][filtnum] = val_loss[2]
                cc_mean_val_1D[count] = val_loss[2]

                # predict on test data
                predicted_batch = model.predict(test_dataset) 
                test_batch = load_nifti(TEST_SUBIDS, rootpath=DATA_PATH,
                                            labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM)
                save_prediction(predicted_batch=predicted_batch, rootpath=DATA_PATH, outpath=OUT_PATH, labelname=TASK_FILE_NAME, labelnum=TASK_FILE_NUM, 
                template_subID=TEST_SUBIDS[0], subIDs=TEST_SUBIDS)
                cc = act_pred_corr(predicted_batch,test_batch)
                
                print(np.mean(np.diagonal(cc)))
               # cc_norm = normalize(cc,axis=0)
               # cc_norm = normalize(cc_norm,axis=1)
               # plt.subplot(1, 2, 1)
               # plt.imshow(cc,cmap="jet")
               # plt.colorbar()
               # plt.subplot(1, 2, 2)
               # plt.imshow(cc_norm,cmap="jet")
               # plt.colorbar()
               # plt.show()

print(cc_mean_test_1D)
print(cc_mean_val_1D)
#with open('/media/Drobo_HCP/HCP_Data/Volume/cc_param_test.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump([cc_mean_test_1D, cc_mean_1D_val], f)
