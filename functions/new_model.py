from functions.old_model import read_test_data, read_train_data
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import shutil 
import time
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.layers import  BatchNormalization, Activation

def compile_model_2(model):
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

def model_fn_2(layers, rates):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                                  activation= 'relu',
                                  input_shape=(128, 128, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(rates))
    model.add(keras.layers.Conv2D(64, (3, 3), activation= 'relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rates))
    for i, nodes in enumerate(layers):
      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(nodes, activation='relu'))
      model.add(BatchNormalization())
      model.add(keras.layers.LeakyReLU(alpha=0.1))
      model.add(keras.layers.Dropout(rates))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.summary()
    compile_model_2(model)
    return model

CLASS_SIZE = 5
FILE_PATH = 'cp-{epoch:04d}.h5'
RETINOPATHY_MODEL = 'retinopathy.h5'
X_train, y_train = read_train_data()
X_test, y_test = read_test_data()


def run_3(X_train = X_train, 
        X_test = X_test,
        y_train = y_train,
        y_test = y_test,
        num_epochs=20,  # Maximum number of epochs on which to train
        train_batch_size=128,  # Batch size for training steps
        job_dir='jobdir', # Local dir to write checkpoints and export model
        checkpoint_epochs='epoch',  #  Save checkpoint every epoch
        load_previous_model=False):
    
    # tf.keras.backend.clear_session()

    try:
        os.makedirs(job_dir)
    except:
        pass

    checkpoint_path = FILE_PATH
    checkpoint_path = os.path.join(job_dir, checkpoint_path)

    retinopathy_model = model_fn_2([512,256], 0.3)
    if load_previous_model:
        # Load the previously saved weights
        latest = get_latest(job_dir, overwrite=True)
        retinopathy_model.load_weights(latest)

    # Model checkpoint callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=2,
        save_freq=checkpoint_epochs,
        mode='max')

    # Tensorboard logs callback
    tblog = keras.callbacks.TensorBoard(
        log_dir=os.path.join(job_dir, 'logs'),
        histogram_freq=0,
        update_freq='epoch',
        write_graph=True,
        embeddings_freq=0)

    callbacks = [checkpoint, tblog]

    # [X_train, Y_train] = read_train_data()
    # [X_test, Y_test] = read_test_data()

    # Data augmentation. Other operations are possible.
    # https://keras.io/api/preprocessing/image/#imagedatagenerator-class
    datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

    history = retinopathy_model.fit(
              datagen.flow(X_train, y_train, batch_size=train_batch_size),
              # steps_per_epoch=20,
              epochs=num_epochs,
              callbacks=callbacks,
              # verbose=2,
              validation_data=(X_test, y_test))

    retinopathy_model.save(os.path.join(job_dir, RETINOPATHY_MODEL))

    return history