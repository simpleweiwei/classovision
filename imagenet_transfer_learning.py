#https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/

import glob
import os
from math import ceil, floor
import random
import shutil
import numpy as np



from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

train_dir = r'C:\Data\computer_vision_coursework\Images\face_images\train'
val_dir = r'C:\Data\computer_vision_coursework\Images\face_images\validation'

nTrain = 1548
nVal = 391

datagen = ImageDataGenerator(rescale=1. / 255)

batch_size = 20
n_classes=4
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain, n_classes))

validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal, n_classes))

shuffle=False
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=shuffle)

validation_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=shuffle)

#Then we use model.predict() function to pass the image through the network which gives
# us a 7 x 7 x 512 dimensional Tensor. We reshape the Tensor into a vector.
# Similarly, we find the validation_features.
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
nImages = nTrain
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))

nImages = nVal
i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size: (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nImages:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))

#Create your own model
#We will create a simple feedforward network with a softmax output layer having 3 classes.
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(n_classes, activation='softmax'))

#Train the model
#Training a network in Keras is as simple as calling model.fit() function as we have seen in our earlier tutorials.

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

#Check Performance
#We would like to visualize which images were wrongly classified.

fnames = validation_generator.filenames

ground_truth = validation_generator.classes

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
#idx2label = dict((v, k) for k, v in label2index.iteritems())
idx2label = dict((label2index[k], k) for k in label2index.keys())

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors), nVal))

#Let us see which images were predicted wrongly
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img

for i in range(len(errors)):
#for i in range(30):
    pred_class = np.argmax(prob[errors[i]])
    pred_label = idx2label[pred_class]
    lab='Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        prob[errors[i]][pred_class])

    print(lab)

    original = load_img('{}/{}'.format(val_dir, fnames[errors[i]]))
    fig = plt.figure()
    plt.imshow(original)
    plt.title(lab)
    fig.show()


print('done!')