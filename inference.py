from PIL import Image
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from tensorflow.keras.models import load_model


# Function to get first frame from a video using OpenCV
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame


if __name__ == '__main__':
    # Read image
    img = r'/Users/avikram/Projects/av-emotion-recognition/dataset/res_img.jpg'
    img = face_recognition.load_image_file(img)

    # emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
    model = load_model(r"./weights/AVER_model.hdf5")
    # Convert image to black and white
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_image = img.reshape(1, 48, 48, 1)
    # Convert img data type to float32
    face_image = face_image.astype('float32')
    predicted_class = np.argmax(model.predict(face_image))
    print(predicted_class)
