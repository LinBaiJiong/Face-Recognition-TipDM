import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import cv2, time, random
from tensorflow.contrib.layers import xavier_initializer
from sklearn.model_selection import train_test_split

def weight_varible(shape):
    initializer=xavier_initializer()
    return tf.Variable(initializer(shape))

def weight_varible_conv2d(shape):
    initializer=xavier_initializer_conv2d()
    return tf.Variable(initializer(shape))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='VALID')

sess = tf.InteractiveSession()

now = int(time.time())
print(now)
tf.set_random_seed(now)
np.random.seed(now)

n_classes = 10

# conv layer-1
x = tf.placeholder(tf.float32, [None, 64, 64], name='x')
y_ = tf.placeholder(tf.float32, [None, n_classes])
x_image = tf.reshape(x, [-1, 64, 64, 1])

W_conv1 = weight_varible([3, 3, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# conv layer-3
W_conv3 = weight_varible([3, 3, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = conv2d(h_pool2, W_conv3) + b_conv3
h_conv3 = tf.nn.relu(h_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# full connection
W_fc1 = weight_varible([6*6*64, 120])
b_fc1 = bias_variable([120])

h_pool3_flat = tf.reshape(h_pool3, [-1, 6*6*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1, name='feature')

# embeddings = tf.nn.l2_normalize(h_fc1, 1, 1e-10, name='embeddings')

# dropout
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([120, n_classes])
b_fc2 = bias_variable([n_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='pred')

# model training
cross_entropy =tf.reduce_mean(-tf.reduce_sum ( y_ * tf.log ( tf.clip_by_value(y_conv, 1e-8,1.0 )) ,1) )

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_p = tf.argmax(y_conv, 1)

sess.run(tf.global_variables_initializer())

train_image = np.load('../result/face_data/train_data.npy')
train_label = np.load('../result/face_data/train_label.npy')
valid_image = np.load('../result/face_data/test_data.npy')
valid_label = np.load('../result/face_data/test_label.npy')

saver = tf.train.Saver()
model_filename = "../result/model/model.ckpt"

size = train_image.shape[0]
for j in range(5):
    index = np.arange(size)
    np.random.shuffle(index)
    for i in range(int(size/20.0)):
        batch_x = train_image[index[i*20:(i+1)*20],:,:]
        batch_y = train_label[index[i*20:(i+1)*20],:]
        if (i%10==0):
            train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict = {x:batch_x, y_:batch_y,keep_prob:1.0})
            test_accuacy, y_pred, test_loss = sess.run([accuracy, y_p, cross_entropy], feed_dict={x:valid_image, y_: valid_label,keep_prob:1.0})
            print("------step %d, train_acc %g, train_loss %g, test_acc %g, test_loss %g"
                %(i, train_acc,train_loss,test_accuacy,test_loss))
        
        train_step.run(feed_dict = {x: batch_x, y_: batch_y,keep_prob:1.0})
        
    train_acc, train_loss = sess.run([accuracy, cross_entropy], feed_dict = {x:batch_x, y_:batch_y,keep_prob:1.0})
    # print (train_loss)
    test_accuacy, y_pred, test_loss = sess.run([accuracy, y_p, cross_entropy], feed_dict={x: valid_image, y_: valid_label,keep_prob:1.0})
    print("step %d, train_acc %g, train_loss %g, test_acc %g, test_loss %g"
        %(j, train_acc,train_loss,test_accuacy,test_loss))

    saver.save(sess, model_filename)