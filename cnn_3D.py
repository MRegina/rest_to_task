"""
Created on Dec 28 2019
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import tensorflow as tf
from datetime import datetime
import os

from utils import NiiSequence, CorrelationMetric, CorrelationLoss

DATA_PATH = 'CNN/'
TRAIN_SUBIDS = ['100307', '101915', '103414']
VALID_SUBIDS = ['103818']
TEST_SUBIDS = ['106319']
REST_FILE_NAME = '_DR2_nosmoothing.nii.gz'
TASK_FILE_NAME = '_motor1.nii.gz'

# otput paths
LOGDIR = os.path.join("logs\\regress_3D", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(os.path.join(LOGDIR, "checkpoints"), exist_ok=False)
CHECKPOINT_PATH = os.path.join(LOGDIR, "checkpoints", "cp-{epoch:04d}.ckpt")

BATCH_SIZE = 1
INPUT_SHAPE = (91, 109, 91, 32)
OUTPUT_SIZE = (91, 109, 91, 1)

# hyperparameters
LEARNING_RATE = 0.001
EPOCH = 100
DROPOUT_RATE = None  # 0.1

# architecture parameters
filter_num1 = 32  # 96
filter_num2 = 32  # 64
filter_num3 = 32  # 32

# tf.config.experimental_run_functions_eagerly(True)

# create datasets
train_dataset = NiiSequence(TRAIN_SUBIDS, shuffle=True, rootpath=DATA_PATH, dataname=REST_FILE_NAME,
                            labelname=TASK_FILE_NAME, batch_size=BATCH_SIZE)
valid_dataset = NiiSequence(VALID_SUBIDS, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME,
                            batch_size=BATCH_SIZE)
test_dataset = NiiSequence(TEST_SUBIDS, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME,
                           batch_size=BATCH_SIZE)


def separable_convolution_3D(x, kernel_size, filter_num, name, activation='relu', dropout_rate=None):
    conv1 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(kernel_size, 1, 1), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv1')(x)
    conv2 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, kernel_size, 1), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv2')(conv1)
    conv3 = tf.keras.layers.Conv3D(filters=filter_num, kernel_size=(1, 1, kernel_size), strides=(1, 1, 1),
                                   padding='same', kernel_initializer='he_uniform', activation=activation,
                                   name=name + 'conv3')(conv2)
    if dropout_rate is not None:
        conv3 = tf.keras.layers.SpatialDropout3D(rate=dropout_rate, name=name + 'dropout')(conv3)
    return conv3


# create model
input = tf.keras.Input(shape=INPUT_SHAPE, name='input')

block1 = separable_convolution_3D(input, kernel_size=5, filter_num=filter_num1, name='block1_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)
block2 = separable_convolution_3D(block1, kernel_size=3, filter_num=filter_num2, name='block2_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)
block3 = separable_convolution_3D(block2, kernel_size=3, filter_num=filter_num3, name='block3_',
                                  activation='relu', dropout_rate=DROPOUT_RATE)

output = separable_convolution_3D(block3, kernel_size=3, filter_num=1, name='block4_',
                                  activation=None, dropout_rate=DROPOUT_RATE)
model = tf.keras.Model(inputs=[input], outputs=[output])

# create callbacks
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=False, period=5)

# compile model for training
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              # Loss function to minimize
              loss=tf.keras.losses.MeanSquaredError(),
              # List of metrics to monitor
              metrics=[tf.keras.losses.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(), CorrelationMetric()])
model.summary()

# training loop
model.fit(train_dataset, epochs=EPOCH, validation_data=valid_dataset, callbacks=[tensorboard_callback, cp_callback])

# evaluate on the test set
test_loss = model.evaluate(test_dataset)
print(test_loss)
