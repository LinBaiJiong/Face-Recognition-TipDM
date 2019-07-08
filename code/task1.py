import cv2, os
from tqdm import tqdm

directory = '../result/faceImages/'
total_person = 10
total_data_per_person = 600

if not os.path.exists(directory):
    os.makedirs(directory)

print('-'*30, '\n')
print('Photo capture begins, the image storage path is {} ...'.format(directory))
for i in range(total_person):
    name = input('Please enter your name: ')

    name_directory = os.path.join(directory, name)
    if not os.path.exists(name_directory):
        os.makedirs(name_directory)

    print('Start the camera ...')
    cap = cv2.VideoCapture(0)
    print('Camera has started ...')

    print('Press the q key on the pop-up window and save the screenshot ...')
    pbar = tqdm(total=total_data_per_person)
    k = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow("capture", frame)     

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(os.path.join(name_directory, '{}.jpg'.format(k)), frame)
            k += 1
            pbar.update(1) 
            if k == total_data_per_person:
                break
    pbar.close()
    print('Photo capture is complete, close the camera ...')
    cap.release()
    cv2.destroyAllWindows()
    print('-'*30, '\n')

print('All photos of {} persons were collected.'.format(total_person))