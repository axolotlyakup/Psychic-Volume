import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Volume range for macOS (0 to 100)
volMin, volMax = 0, 100

def set_system_volume(volume):
    """Set the system volume on macOS using osascript."""
    osascript_command = f"osascript -e 'set volume output volume {int(volume)}'"
    os.system(osascript_command)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        # Convert hand gesture length to volume level
        vol = np.interp(length, [15, 220], [volMin, volMax])
        print(f"Volume: {vol}, Length: {length}")
        set_system_volume(vol)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
