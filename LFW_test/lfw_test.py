import numpy as np
import os
import tensorflow as tf
import cv2, pprint
from mtcnn.mtcnn import MTCNN
from tqdm import tqdm
import matplotlib.pylab as plt

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)
    
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)
        
def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return path_list, issame_list
    
pairs = read_pairs('./pairs.txt')
lfw_paths, actual_issame = get_paths('./lfw/', pairs)

model_dir = "../result/model/"
ckpt = tf.train.get_checkpoint_state(model_dir)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)

img_x = sess.graph.get_tensor_by_name('x:0')
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
pred = sess.graph.get_tensor_by_name('pred:0')
feature = sess.graph.get_tensor_by_name('feature:0')

detector = MTCNN()

def preprocess_image(img_dir, detector, size=64):
    img = cv2.imread(img_dir)
    detected = detector.detect_faces(img)
    if (len(detected) == 1) :
        x, y, w, h = detected[0]['box']
        x = max(x, 0)
        y = max(y, 0)
        face = img[y:y+h, x:x+w]
        im = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        im_resize = cv2.resize(im, (size, size))
        im_resize = im_resize.reshape([1, size, size])
        return im_resize / 255.
    else:
        return np.zeros([2, 1])
def norm(feature1, feature2):
    return np.sqrt(np.sum(np.square(np.tile(feature1, (feature2.shape[0], 1)) - feature2), axis=1))
    
acc = 0
sim_list = np.zeros([4827, 2])
k = 0
for i in tqdm(range(6000)):
    img1 = preprocess_image(lfw_paths[2*i], detector)
    img2 = preprocess_image(lfw_paths[2*i+1], detector)
    if ((img1.shape[0] != 2) and (img2.shape[0] != 2)):
        img1_feature = sess.run(feature, feed_dict={img_x:img1, keep_prob:1.0})
        img2_feature = sess.run(feature, feed_dict={img_x:img2, keep_prob:1.0})
        
        sim = norm(img1_feature, img2_feature)
        sim_list[k, 0] = sim
        sim_list[k, 1] = int(actual_issame[i])
        k += 1

th = np.linspace(0, int(np.max(sim_list[:,0])), 1000)
acc_list = np.zeros([th.shape[0]])
for i in range(th.shape[0]):
    t_index = np.argwhere(sim_list[:,1]==1)
    f_index = np.argwhere(sim_list[:,1]==0)
    tp = np.argwhere(sim_list[t_index,0] < th[i]).shape[0]
    fp = np.argwhere(sim_list[f_index,0] >= th[i]).shape[0]
    acc = (tp + fp) / sim_list.shape[0]
    acc_list[i] = acc

    
plt.figure(figsize=[8,5])
plt.plot(th, acc_list,linewidth=3)
plt.tick_params(labelsize=15)
font2 = {'weight':'normal','size':15}
plt.xlabel('Threshold',font2)
plt.ylabel('Accuracy',font2)
plt.savefig('lfw_acc.eps')
plt.show()