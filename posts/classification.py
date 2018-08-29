import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


class Image_identify():

    def process_test_data(test_data):
        IMG_SIZE = 50
        LR = 1e-3
        testing_data = []
        list=[]
        for img in tqdm(os.listdir(test_data)):
            list.append(img)
        if img != '.DS_Store':
            img = list[0]
            path = os.path.join(test_data, img)
            img_num = img.split('.')[-1]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        test_data = testing_data
        fig = plt.figure()
        for num, data in enumerate(test_data):
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)

            tf.reset_default_graph()

            convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

            convnet = conv_2d(convnet, 32, 5, activation='relu')
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 64, 5, activation='relu')
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 128, 5, activation='relu')
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 64, 5, activation='relu')
            convnet = max_pool_2d(convnet, 5)

            convnet = conv_2d(convnet, 32, 5, activation='relu')
            convnet = max_pool_2d(convnet, 5)

            convnet = fully_connected(convnet, 1024, activation='relu')
            convnet = dropout(convnet, 0.8)

            convnet = fully_connected(convnet, 2, activation='softmax')
            convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                                 name='targets')

            model_import = tflearn.DNN(convnet, tensorboard_dir='log')

            model_import.load("./MODEL_NAME", weights_only=False)

            model_out = model_import.predict([data])[0]

            if np.argmax(model_out) == 1:
                str_label = 'disease'
            else:
                str_label = 'healthy'
            y.imshow(orig, cmap='gray')
            plt.title(str_label)
            y.axes.get_xaxis().set_visible(False)
            y.axes.get_yaxis().set_visible(False)
        plt.show()
        plt.show()
        return str_label