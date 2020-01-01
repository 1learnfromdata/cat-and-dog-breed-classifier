# -*- coding: utf-8 -*-

import os
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as Image

from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

dataset_path_new = "cats_and_dogs_filtered/"  # download cat and dog image data from Kaggle.com
train_dir = os.path.join(dataset_path_new, "train")
validation_dir = os.path.join(dataset_path_new, "validation")

IMG_SHAPE = (128, 128, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights="imagenet")

base_model.summary()

base_model.trainable = False
base_model.output
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

prediction_layer = tf.keras.layers.Dense(units=1,
                                      activation='sigmoid')(global_average_layer)

model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
           loss="binary_crossentropy",
           metrics=["accuracy"])

# data_gen_train = ImageDataGenerator(rescale=1/255.)
data_gen_valid = ImageDataGenerator(rescale=1/255.)

data_gen_train = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        rescale=1/255.)

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.001)

train_generator = data_gen_train.flow_from_directory(train_dir,
                                                     target_size=(128,128),
                                                     batch_size=64,
                                                     class_mode="binary")

valid_generator = data_gen_valid.flow_from_directory(validation_dir,
                                                     target_size=(128,128),
                                                     batch_size=64,
                                                     class_mode="binary")
                                                                                                         # Fit the model

model.fit_generator(train_generator,
                    epochs=40,
                    validation_data=valid_generator,
                                        callbacks=[learning_rate_reduction])

valid_loss, valid_accuracy = model.evaluate_generator(valid_generator)
model_json = model.to_json()
with open("catanddog_model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("catanddog_model.h5")
print("Accuracy after transfer learning: {}".format(valid_accuracy))

# =============================================================================
# IMAGE_SHAPE_2 = (128, 128)
#
# grace_hopper = Image.open('adorable-animal-cat-320014.jpg').resize(IMAGE_SHAPE_2)
# grace_hopper = np.array(grace_hopper)/255.0
# grace_hopper.shape
# result = model.predict(grace_hopper[np.newaxis, ...])
# result[0][0].round()
# =============================================================================

