import os
import cv2
import torch
import numpy as np
from pygame import mixer
from model import DrowsinessDetectionCNN

# ---------------------------
# Configurations
# ---------------------------
HAAR_FACE = 'haar_cascade_files/haarcascade_frontalface_alt.xml'
HAAR_LEYE = 'haar_cascade_files/haarcascade_lefteye_2splits.xml'
HAAR_REYE = 'haar_cascade_files/haarcascade_righteye_2splits.xml'
ALARM_SOUND = 'alarm.wav'
MODEL_PATH = 'models/drowsiness_model.pth'

SCORE_THRESHOLD = 15  # frames

# ---------------------------
# Error checks
# ---------------------------
for f in [HAAR_FACE, HAAR_LEYE, HAAR_REYE, ALARM_SOUND, MODEL_PATH]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Required file not found: {f}")

# ---------------------------
# Initialize
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(HAAR_FACE)
leye_cascade = cv2.CascadeClassifier(HAAR_LEYE)
reye_cascade = cv2.CascadeClassifier(HAAR_REYE)

# Initialize alarm
mixer.init()
sound = mixer.Sound(ALARM_SOUND)

# Load model
model = DrowsinessDetectionCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------------------
# Detect model input size
# ---------------------------
with torch.no_grad():
    dummy = torch.zeros(1, 3, 128, 128).to(device)  # trial size
    try:
        model(dummy)
        FRAME_WIDTH = FRAME_HEIGHT = 128
    except RuntimeError as e:
        # Try increasing/decreasing until it works
        found = False
        for size in range(32, 513, 8):  # check from 32 to 512 px
            try:
                test = torch.zeros(1, 3, size, size).to(device)
                model(test)
                FRAME_WIDTH = FRAME_HEIGHT = size
                found = True
                break
            except RuntimeError:
                continue
        if not found:
            raise RuntimeError("Could not auto-detect correct input size.")

print(f"✅ Model expects {FRAME_WIDTH}×{FRAME_HEIGHT} images.")

# ---------------------------
# Helper function: predict eye state
# ---------------------------
def predict_eye(eye_frame):
    eye_resized = cv2.resize(eye_frame, (FRAME_WIDTH, FRAME_HEIGHT))
    eye_rgb = cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB)
    eye_rgb = eye_rgb / 255.0
    tensor_eye = torch.FloatTensor(eye_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor_eye)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    return pred  # 0 = Closed, 1 = Open

# ---------------------------
# Main loop
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25,25))
    left_eyes = leye_cascade.detectMultiScale(gray)
    right_eyes = reye_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,100,100), 1)

    for (x,y,w,h) in right_eyes:
        rpred[0] = predict_eye(frame[y:y+h, x:x+w])
        break

    for (x,y,w,h) in left_eyes:
        lpred[0] = predict_eye(frame[y:y+h, x:x+w])
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        status = "Closed"
    else:
        score -= 1
        status = "Open"

    if score < 0:
        score = 0

    cv2.putText(frame, f'{status}', (10,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Score:{score}', (100,height-20), font, 1, (255,255,255), 1, cv2.LINE_AA)

    if score > SCORE_THRESHOLD:
        try:
            sound.play()
        except:
            pass
        thicc = thicc + 2 if thicc < 16 else 2
        cv2.rectangle(frame, (0,0), (width,height), (0,0,255), thicc)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
