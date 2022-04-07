import cv2 as cv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from tensorflow.keras.models import load_model


# Function to get first frame from a video using OpenCV
def get_first_frame(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame


if __name__ == '__main__':
    # Read image
    img = get_first_frame('/home/pi/Desktop/video.mp4')
    image = face_recognition.load_image_file("../test_images/040wrmpyTF5l.jpg")

    # Recognize face in the image
    face_locations = face_recognition.face_locations(img)

    # Create bounding box
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]

    # Load trained model
    model = load_model("./emotion_detector_models/model.hdf5")
    predicted_class = np.argmax(model.predict(face_image))

