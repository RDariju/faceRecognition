import cv2
import time
from datetime import datetime
import pyttsx3

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize face recognizer and load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

# List of names corresponding to labels
name_list = ["", "Ravindu"]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to determine the greeting based on the current time
def get_greeting():
    current_hour = datetime.now().hour
    if current_hour < 12:
        return "Good morning"
    elif current_hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

# Function to speak the greeting
def speak_greeting(text):
    engine.setProperty('rate', 150)  # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    # Set male voice (index might differ depending on your system)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Typically, index 0 is for a male voice

    engine.say(text)
    engine.runAndWait()

# Variables to keep track of the last recognized person and time
last_serial = None
last_announcement_time = time.time()

# Announcement interval (in seconds)
announcement_interval = 10

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        greeting = get_greeting()

        if conf < 50:
            name = name_list[serial]
            text = f"{greeting}, {name}"
        else:
            text = greeting

        # Draw rectangle around face and display the greeting
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Get the current time
        current_time = time.time()

        # Check if the face has changed or 30 seconds have passed since the last announcement
        if serial != last_serial or (current_time - last_announcement_time) > announcement_interval:
            speak_greeting(text)
            last_serial = serial
            last_announcement_time = current_time

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
