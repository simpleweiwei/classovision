'''VGGFace models for Keras.
# Reference:
- [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- [VGGFace2: A dataset for recognising faces across pose and age](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/vggface2.pdf)
'''

from __future__ import print_function


import os
import glob
import numpy as np

from keras import models
from keras import layers
from keras import optimizers

from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras.preprocessing.image import ImageDataGenerator

def VGGFace(include_top=True, model='vgg16', weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=None):
    #https://github.com/rcmalli/keras-vggface/blob/master/keras_vggface/vggface.py
    """Instantiates the VGGFace architectures.
    Optionally loads weights pre-trained
    on VGGFace datasets. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "vggface" (pre-training on VGGFACE datasets).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        model: selects the one of the available architectures
            vgg16, resnet50 or senet50 default is vgg16.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    from keras_vggface.models import RESNET50, VGG16, SENET50

    if weights not in {'vggface', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `vggface`'
                         '(pre-training on VGGFace Datasets).')

    if model == 'vgg16':

        if classes is None:
            classes = 2622

        if weights == 'vggface' and include_top and classes != 2622:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 2622')

        return VGG16(include_top=include_top, input_tensor=input_tensor,
                     input_shape=input_shape, pooling=pooling,
                     weights=weights,
                     classes=classes)


    if model == 'resnet50':

        if classes is None:
            classes = 8631

        if weights == 'vggface' and include_top and classes != 8631:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 8631')

        return RESNET50(include_top=include_top, input_tensor=input_tensor,
                        input_shape=input_shape, pooling=pooling,
                        weights=weights,
                        classes=classes)

    if model == 'senet50':

        if classes is None:
            classes = 8631

        if weights == 'vggface' and include_top and classes != 8631:
            raise ValueError(
                'If using `weights` as vggface original with `include_top`'
                ' as true, `classes` should be 8631')

        return SENET50(include_top=include_top, input_tensor=input_tensor,
                        input_shape=input_shape, pooling=pooling,
                        weights=weights,
                        classes=classes)

def gen_model_using_keras_vggface(n_classes):

    #load base model from https://github.com/rcmalli/keras-vggface
    vgg_model=VGGFace(include_top=False, model='vgg16', weights='vggface',
            input_shape=(224,224,3), pooling=None,
            classes=None)

    # Freeze the layers except the last 4 layers
    for layer in vgg_model.layers[:-4]:
        layer.trainable = False

    # Check the trainable status of the individual layers
    #for layer in vgg_model.layers:
    #    print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_model)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


if __name__ == '__main__':

    gen_new_model=True
    model_save_path = r'saved_networks\vgg_face61.h5'
    train_dir = r'C:\Data\computer_vision_coursework\Images\face_images\from_group_photos\train'
    val_dir = r'C:\Data\computer_vision_coursework\Images\face_images\from_group_photos\val'


    if gen_new_model:
        model = gen_model_using_keras_vggface(61)
        model.save(model_save_path)
    else:
        model = models.load_model(model_save_path)

    #now get data for training and validation
    image_size=224
    train_glob=os.path.join(train_dir,r'*\*.jpg')
    train_files=glob.glob(train_glob)

    val_glob=os.path.join(val_dir,r'*\*.jpg')
    val_files=glob.glob(val_glob)

    nTrain = len(train_files)
    nVal = len(val_files)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # Change the batchsize according to your system RAM
    train_batchsize = 20
    val_batchsize = 10

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='categorical',
        shuffle=False)

    # Train the model
    # Training a network in Keras is as simple as calling model.fit() function as we have seen in our earlier tutorials.


    # Train the model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / train_generator.batch_size,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)

    # Save the model
    if not os.path.isdir(os.path.dirname(model_save_path)):
        os.mkdir(os.path.dirname(model_save_path))

    save_path=model_save_path.replace('.h5','_trained.h5')
    model.save(save_path)

    print('Done!')
