import face_recognition
import numpy as np
import cv2
from keras.models import load_model

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

# image = face_recognition.load_image_file("./1.jpg");
image = face_recognition.load_image_file("./2.jpeg");

face_locations = face_recognition.face_locations(image);

top, right, bottom, left = face_locations[0]

face_image = image[top:bottom, left:right]

face_image = cv2.resize(face_image, (48,48))

face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

model = load_model("./model_v6_23.hdf5")

predicted_class = np.argmax(model.predict(face_image))
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)

