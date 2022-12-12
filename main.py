import numpy
import time
from mtcnn import MTCNN
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import sys
import cv2

def face_detection(url):
    start_time = time.time()
    img = cv2.cvtColor(cv2.imread(url), cv2.COLOR_BGR2RGB)
    detector = MTCNN()  # Multi Convolutional Neural Network library - mtcnn
    detect = detector.detect_faces(img)
    for i in range(len(detect)):
        face_box = detect[i]['box']
        plt.gca().add_patch(Rectangle(face_box[:2], face_box[2], face_box[3],
                                  edgecolor='red',
                                  facecolor='none',
                                  lw=2))
    plt.imshow(img)
    plt.title(f'found {i} faces within {round(time.time()-start_time,2)} seconds')
    plt.show()
    return 0

if __name__ == '__main__':
    face_detection("manyfaces.jpg")
