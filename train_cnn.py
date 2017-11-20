# A practice for CNN, study a github project for simpson character identification

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
from random import shuffle
from collections import Counter

from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam


# map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson',
#         3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel',
#         7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson',
#         11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak',
#         14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

map_characters = {0: 'tree_1', 1: 'tree_2'}

pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)
pictures_per_class = 1000
test_size = 0.15


# load_pictures(): load pictures and labels from the characters folder
def load_pictures(BGR):
    """
    Load pictures from folders for characters from the map_characters dict and create a numpy dataset and
    a numpy labels set. Pictures are re-sized into picture_size square.
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: dataset, labels set
    """
    pics = []
    labels = []

    # for loops: https://wiki.python.org/moin/ForLoop
    # dict.items(): \
    # https://stackoverflow.com/questions/10458437/what-is-the-difference-between-dict-items-and-dict-iteritems
    # this for loop is used for: the traversal of the map_characters, k is the key and char is the value
    for k, char in map_characters.items():
        # print(k, char)

        # glob module: https://docs.python.org/3.5/library/glob.html
        # k for k in x (List Comprehensions): https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions
        # picture is a array of the picture names in target folder
        # pictures = [k for k in glob.glob('./characters/%s/*' % char)]
        pictures = [k for k in glob.glob('./frames/%s/*' % char)]
        # print(pictures)

        # nb_pic: the number of the pictures array
        # https://stackoverflow.com/questions/2529536/python-idiom-for-if-else-expression
        nb_pic = round(pictures_per_class/(1-test_size)) if round(pictures_per_class/(1-test_size))<len(pictures) else len(pictures)
        # nb_pic = len(pictures)
        # print(nb_pic)

        # np.random.choice(pictures, nb_pic): use nb_pic to randomly generate a \
        # array consists of the nb_picture elements of pictures
        # print(np.random.choice(pictures, nb_pic))
        # np.random.choice(): https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html
        # pic is the relative path of an image
        for pic in np.random.choice(pictures, nb_pic):
            # cv2.imread(): read the image
            # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html
            # a.shape is (width, height, colour?)
            a = cv2.imread(pic)
            if BGR:
                a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

            # resize every image to pic_size*pic_size
            a = cv2.resize(a, (pic_size, pic_size))

            # add a to pics array
            # add k to label array, k is from 0 to 17
            pics.append(a)
            labels.append(k)

    return np.array(pics), np.array(labels)


def get_dataset(save=False, load=False, BGR=False):
    """
    get X_train, X_test, y_train, y_test

    Create the actual dataset split into train and test, pictures content is as float32 and
    normalized (/255.). The dataset could be saved or loaded from h5 files.
    :param save: saving or not the created dataset
    :param load: loading or not the dataset
    :param BGR: boolean to use true color for the picture (RGB instead of BGR for plt)
    :return: X_train, X_test, y_train, y_test (numpy arrays)
    """
    if load:
        # load data from h5py file

        # open and read the training and testing image data from dataset.h5
        h5f = h5py.File('dataset.h5', 'r')
        # Python slice notation: https://stackoverflow.com/questions/509211/explain-slice-notation
        # a[:]: a copy of the whole array
        X_train = h5f['X_train'][:]
        X_test = h5f['X_test'][:]
        #close the file
        h5f.close()

        # open and read the training and testing label data from labels.h5
        h5f = h5py.File('labels.h5', 'r')
        y_train = h5f['y_train'][:]
        y_test = h5f['y_test'][:]
        h5f.close()
    else:
        # load data from image folder

        # X is picture array of the training set, y is the labels array of the training set
        X, y = load_pictures(BGR)
        # to_categorical: https://keras.io/utils/#to_categorical
        # Converts a class vector (integers) y to binary class matrix.
        y = keras.utils.to_categorical(y, num_classes)
        # train_test_split: Split arrays or matrices into random train and test subsets
        # X_train and x_test are split from X, y_train and y_test are split from y
        # if the type of the test_size is float \
        # it represents the proportion of the dataset to include in the test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        if save:
            # save X_train, X_test to the dataset.h5
            h5f = h5py.File('dataset.h5', 'w')
            h5f.create_dataset('X_train', data=X_train)
            h5f.create_dataset('X_test', data=X_test)
            h5f.close()

            # y_train, y_test to the labels.h5
            h5f = h5py.File('labels.h5', 'w')
            h5f.create_dataset('y_train', data=y_train)
            h5f.create_dataset('y_test', data=y_test)
            h5f.close()

    # Feature normalisation of X_train and X_test
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)

    if not load:
        dist = {
        k: tuple(d[k] for d in [dict(Counter(np.where(y_train == 1)[1])), dict(Counter(np.where(y_test == 1)[1]))])
        for k in range(num_classes)}

        # print the number of the train picture and test picture of each character
        print('\n'.join(["%s : %d train pictures & %d test pictures" % (map_characters[k], v[0], v[1])
                         for k, v in sorted(dist.items(), key=lambda x: x[1][0], reverse=True)]))

    return X_train, X_test, y_train, y_test


def create_model_two_conv(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(512))
    model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    # Stochastic gradient descent optimizer
    opt = SGD(lr=0.01, decay=5e-5, momentum=0.9, nesterov=True)

    return model, opt


def create_model_four_conv(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    # Activations can either be used through an Activation layer, \
    # or through the activation argument supported by all forward layers:
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt


def create_model_six_conv(input_shape):
    """
    CNN Keras model with 6 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Stochastic gradient descent optimizer
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return model, opt


def load_model_from_checkpoint(weights_path, six_conv=False, input_shape=(pic_size, pic_size, 3)):
    """
    Choose create model method, four_conv or six_conv
    load weights from weights_path
    Configures the learning process, and return the compiled model

    :param weights_path:
    :param six_conv:
    :param input_shape:
    :return:
    """
    if six_conv:
        model, opt = create_model_six_conv(input_shape)
    else:
        model, opt = create_model_four_conv(input_shape)

    model.load_weights(weights_path)

    # model.compile(): Configures the learning process.
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

    return model


def lr_schedule(epoch):
    """
    learning rate schedule for each iteration
    :param epoch:
    :return:
    """
    lr = 0.01
    return lr*(0.1**int(epoch/10))


def training(model, X_train, X_test, y_train, y_test, data_augmentation=True):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for data_augmentation (default:True)
    :param callback: boolean for saving model checkpoints and get the best saved model
    :param six_conv: boolean for using the 6 convs model (default:False, so 4 convs)
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epoch)
    """
    if data_augmentation:
        # data augmentation, image augmentation:
        # https://medium.com/towards-data-science/image-augmentation-for-deep-learning-histogram-equalization-a71387f609b2
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
        filepath="weights_6conv_%s.hdf5" % time.strftime("%d%m/%Y")
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(lr_schedule), checkpoint]

        # model.fit(): Trains the model for a fixed number of epochs.
        # model.fit_generator(): Fits the model on data generated batch-by-batch by a Python generator.
        # https://keras.io/models/sequential/
        # history = model.fit_generator(datagen.flow(X_train, y_train,
        #                             batch_size=batch_size),
        #                             steps_per_epoch=X_train.shape[0] // batch_size,
        #                             epochs=40,
        #                             validation_data=(X_test, y_test),
        #                             callbacks=callbacks_list)

        history = model.fit_generator(datagen.flow(X_train, y_train,
                                                   batch_size=batch_size),
                                      steps_per_epoch=X_train.shape[0] // batch_size,
                                      epochs=40,
                                      validation_data=(X_test, y_test),
                                      callbacks=None)
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
    return model, history


# https://www.zhihu.com/question/49136398
# https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == '__main__':
    X_train, X_test, y_train, y_test = train_cnn.get_dataset(load=True)
    model, opt = train_cnn.create_model_six_conv(X_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    model, history = train_cnn.training(model, X_train, X_test, y_train, y_test, data_augmentation=True)
