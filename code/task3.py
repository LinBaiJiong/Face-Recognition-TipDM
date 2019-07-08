import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import cv2, time, random
from tensorflow.contrib.layers import xavier_initializer
from sklearn.model_selection import train_test_split

n_classes = 10

image = np.zeros([n_classes*600, 64, 64])
label = np.zeros([n_classes*600, n_classes])
for c, name in enumerate(['wuyuhao', 'wutian', 'liziying', 'yuwoliang','liangzhiming',
                          'zengsi', 'luolineng', 'niruxing', 'hedelin', 'laojunhao']):
    for i in range(600):
        im = cv2.imread('../result/faceImageGray/{}/{}.jpg'.format(name, i))
        image[c*600+i,:,:] = cv2.resize(im[:,:,0:1], (64, 64)) / 255.
        label[c*600+i, c] = 1

train_image,valid_image, train_label, valid_label = train_test_split(image,
                                                   label,
                                                   test_size = 0.2,
                                                   random_state = 0,
                                                   shuffle=True)

np.save('../result/face_data/train_data.npy', train_image)
np.save('../result/face_data/train_label.npy', train_label)
np.save('../result/face_data/test_data.npy', valid_image)
np.save('../result/face_data/test_label.npy', valid_label)