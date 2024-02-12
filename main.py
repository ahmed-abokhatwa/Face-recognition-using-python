import cv2
from simple_facerec import SimpleFacerec
import os
import pyttsx3
import time  # Import the time module

os.chdir(r"C:\Users\New\Desktop\Face-Rec")

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Initialize the TTS engine
engine = pyttsx3.init()

# Set a 5-second gap between speaking
engine.setProperty("rate", 150)  # Adjust the speaking rate if needed
engine.setProperty("volume", 0.9)  # Adjust the volume if needed

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        # Speak the detected name with a 5-second gap
        engine.say(f'It\'s {name}')
        engine.runAndWait()

        # Add a 4-second gap between each speak
        time.sleep(4)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
