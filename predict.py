"""
Created on Jan 14 2020
Code for 3D task activation regression with convolutional networks based on resting state connectivity data
@author: mregina
"""

import tensorflow as tf
from utils import NiiSequence, save_prediction

DATA_PATH = 'CNN/'
TRAIN_SUBIDS = ['100307', '101915', '103414']
VALID_SUBIDS = ['103818']
TEST_SUBIDS = ['106319']
REST_FILE_NAME = '_DR2_nosmoothing.nii.gz'
TASK_FILE_NAME = '_motor1.nii.gz'

TRAINED_MODEL_DIR = 'logs\\regress_3D\\20200114-192620\\'
BEST_CHECKPOINT = 100


with open(TRAINED_MODEL_DIR + 'model.json') as model_json:
 model = tf.keras.models.model_from_json(model_json.read())
model.load_weights(TRAINED_MODEL_DIR + 'checkpoints\\' + "cp-%04d.ckpt" % BEST_CHECKPOINT)

# create dataset for prediction
prediction_dataset = NiiSequence(TEST_SUBIDS, rootpath=DATA_PATH, dataname=REST_FILE_NAME, labelname=TASK_FILE_NAME,
                                 batch_size=1)

# predict on test data
predicted_batch = model.predict(prediction_dataset)
save_prediction(predicted_batch=predicted_batch, rootpath=DATA_PATH, labelname=TASK_FILE_NAME,
                template_subID=TEST_SUBIDS[0], subIDs=TEST_SUBIDS, savepath=TRAINED_MODEL_DIR)
