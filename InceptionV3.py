# -*- coding: utf-8 -*-
"""
Created on Thu May 13 20:53:30 2021

@author: yoges
"""

from keras.applications import InceptionV3

conv_base = InceptionV3(weights='imagenet' , include_top = False ,  input_shape=(150, 150, 3))

conv_base.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'D:\DataSet'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

train_datagen = ImageDataGenerator(
       rescale=1./255,
       rotation_range=40,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       fill_mode='nearest')


def extract_features_train(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count))
    generator = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels



def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 3, 3, 2048))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

#Extracting features using the pretrained convolutional base
print('\nExtracting Training Features...')
train_features, train_labels = extract_features_train(train_dir, 2000)
print('\nExtracting Validation Features...')
validation_features, validation_labels = extract_features(validation_dir, 1000)
print('\nExtracting testing Features...')
test_features, test_labels = extract_features(test_dir, 1000)




train_features = np.reshape(train_features, (2000, 3 * 3 * 2048))
validation_features = np.reshape(validation_features, (1000, 3 * 3 * 2048))
test_features = np.reshape(test_features, (1000, 3 * 3 * 2048))

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=3 * 3 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.SGD(lr=0.001 , momentum=0.9 , decay=0.01),
              loss='binary_crossentropy',
              metrics=['acc'])


import tensorflow as tf
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001, #0.01%
    patience=7,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=True)

EPOCHS=40
history = model.fit(train_features, train_labels,
                    epochs=EPOCHS,
                    batch_size=20,
                    callbacks=callback,
                    validation_data=(validation_features, validation_labels))

# show test data results
predictions = model.predict(test_features)


def ShowResults(ActualLbl,PredictedLbl):
    PredictedLbl = np.where(PredictedLbl >= 0.5, 1, PredictedLbl)
    PredictedLbl = np.where(PredictedLbl < 0.5, 0, PredictedLbl)
    from sklearn import metrics
    print('Confusion Matrix:\n',metrics.confusion_matrix(ActualLbl, PredictedLbl))
    #tn, fp, fn, tp = metrics.confusion_matrix(ActualLbl, PredictedLbl).ravel()
    #print(tn,fp,fn,tp)
    #Ac= (tp+tn)/(tn+fp+fn+tp)
    #Sn=tp/(tp+fn) #recall
    #Sp = tn/(tn+fp)
    #Pr = tp/(tp+fp)
    #F1=2*Pr*Sn/(Pr+Sn)    
    #print ('\nAccuracy:',Ac)
    #print ('\nF1 score:',F1)
    
ShowResults(test_labels,predictions)

########################################################################################

# Training is very fast, since we only have to deal with two `Dense` layers -- an epoch takes less than one second even on CPU.
# Let's take a look at the loss and accuracy curves during training:
  

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('validation loss on Patience 5')
# plt.legend()

plt.show()


