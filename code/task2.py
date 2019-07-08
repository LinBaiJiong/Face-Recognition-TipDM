from mtcnn.mtcnn import MTCNN
import cv2, os

directory = '../result/faceImages'
output_directory = '../result/faceImageGray'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

detector = MTCNN()

for name in os.listdir(directory):
    if not os.path.exists(os.path.join(output_directory, name)):
        os.makedirs(os.path.join(output_directory, name))

    for i in range(0, 600):
        file = '{}.jpg'.format(i)
        image = cv2.imread(os.path.join(directory, name, file))

        detected = detector.detect_faces(image)

        if detected:
            box = detected[0]['box']
            # print(box)
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)
            face = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            cv2.rectangle(image, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (0,0,255), 3)

            cv2.imwrite(os.path.join(output_directory, name, file), face_gray)
        else:
            cv2.imwrite(os.path.join(output_directory, name, file), image)