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

from functions.new_model import run_3
from functions.old_model import read_test_data, read_train_data
from functions.preprocessing import preprocess_data
import tensorflow as tf

X_train, y_train = read_train_data()
X_test, y_test = read_test_data()

X_train_proc = preprocess_data(X_train)

history = run_3(load_previous_model=False, num_epochs=1)
