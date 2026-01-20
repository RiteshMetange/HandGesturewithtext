import cv2
import mediapipe as mp
import csv
import os
import time

print("COLLECT_DATA.PY STARTED")
time.sleep(1)

# ---------------- SETTINGS ----------------
DATASET_DIR = "dataset"

GESTURE_KEYS = {
    'h': 'hi',
    'w': 'how',
    'a': 'are',
    'y': 'you',
    'c': 'clear',

    'l': 'hello',
    't': 'thanks',
    's': 'sorry',
    'b': 'bye',
    '/':'Nameste'
}

os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------- MEDIAPIPE INIT ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA INIT ----------------
cap = cv2.VideoCapture(0)
time.sleep(2)

if not cap.isOpened():
    print("ERROR: Camera could not be opened")
    input("Press ENTER to exit")
    exit()

print("Camera opened successfully")

print("\n--- DATA COLLECTION CONTROLS ---")
for k, v in GESTURE_KEYS.items():
    print(f"Press '{k}' to save gesture: {v}")
print("Press 'q' to quit\n")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks_list = None

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            landmarks_list = []

            for lm in hand_lms.landmark:
                landmarks_list.append(lm.x)
                landmarks_list.append(lm.y)

            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS
            )

    # ---------------- UI TEXT ----------------
    cv2.putText(
        frame,
        "Welcome",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Data Collection", frame)

    key = cv2.waitKey(10) & 0xFF

    # Quit
    if key == ord('q'):
        print("Quitting data collection")
        break

    # Save data
    if landmarks_list is not None:
        key_char = chr(key) if key != 255 else None

        if key_char in GESTURE_KEYS:
            gesture_name = GESTURE_KEYS[key_char]
            file_path = os.path.join(DATASET_DIR, f"{gesture_name}.csv")

            with open(file_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(landmarks_list)

            print(f"Saved sample for gesture: {gesture_name}")

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
input("Data collection finished. Press ENTER to exit.")
