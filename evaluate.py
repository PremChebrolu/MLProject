import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('handwritten.model')

#loss, accuracy = model.evaluate(xtest, ytest)