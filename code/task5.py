from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import cv2
import numpy as np

name_list = ['wuyuhao', 'wutian', 'liziying', 'yuwoliang','liangzhiming',
             'zengsi', 'luolineng', 'niruxing', 'hedelin', 'laojunhao']

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
cameraCapture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('out.avi', fourcc, 10, (640, 480))
success, frame = cameraCapture.read()

feature_data = np.zeros([len(name_list), 600, 120])
for k, name in enumerate(name_list):
    for i in range(600):
        im = cv2.imread('../result/faceImageGray/{}/{}.jpg'.format(name, i))
        im_resize = cv2.resize(im[:,:,0:1], (64, 64))
        im_resize = im_resize.reshape([1, 64, 64]) / 255.
        y_feature = sess.run(feature, feed_dict={img_x:im_resize,keep_prob:1.0})

        feature_data[k, i, :] = y_feature

def preprocess_image(img, size=64):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im_resize = cv2.resize(im, (size, size))
    im_resize = im_resize.reshape([1, size, size])
    return im_resize / 255.

def cosine(feature1, feature2):
    return np.dot(feature1,feature2.T)/(np.linalg.norm(feature1)*(np.linalg.norm(feature2, axis=1)))

def norm(feature1, feature2):
    return np.sqrt(np.sum(np.square(np.tile(feature1, (feature2.shape[0], 1)) - feature2), axis=1))

def sim(feature_data, feature, name_list):
    cc = np.zeros([len(name_list)])
    for i in range(feature_data.shape[0]):
        c = np.mean(norm(feature, feature_data[i]))
        cc[i] = c
    if np.min(cc) < 18:
        return name_list[np.argmin(cc)]
    else:
        return 'other'

def show_faces (img, out):
    t1 = cv2.getTickCount()
    detected = detector.detect_faces(img)
    if detected:
        face_name = []
        face_box = []
        for i in range (len(detected)):
            x, y, w, h = detected[i]['box']
            x = max(x, 0)
            y = max(y, 0)
            face_box.append(detected[i]['box'])
            face = img[y:y+h, x:x+w]
            face_preprocess = preprocess_image(face)
            y_feature, y_pred = sess.run([feature, pred], feed_dict={img_x: face_preprocess,keep_prob:1.0})

            name = sim(feature_data, y_feature, name_list)            
            face_name.append(name)

        t2 = cv2.getTickCount()
        t = (t2-t1)/cv2.getTickFrequency()
        fps = 1.0/t
        for i in range(len(detected)):
            x, y, w, h = face_box[i]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = cv2.putText(img,face_name[i],(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            img = cv2.putText(img, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imshow("Camera", img)
    out.write(img)

while success and cv2.waitKey(1) == -1:
    success, frame = cameraCapture.read()
    show_faces(frame, out)

cameraCapture.release() 
out.release() 
cv2.destroyAllWindows() 