
# may be final one 
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, jsonify, request
import threading
import webbrowser

# ---------------- FLASK INIT ----------------
app = Flask(__name__)

# ---------------- GLOBAL STATE ----------------
camera_running = False
current_sentence = ""

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("mp_hand_gesture")

with open("gesture.names", "r") as f:
    class_names = f.read().splitlines()

# ---------------- MEDIAPIPE INIT ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils


# ---------------- CAMERA THREAD ----------------
def camera_loop():
    global camera_running, current_sentence

    cap = cv2.VideoCapture(0)

    sentence_words = []
    previous_word = ""
    cooldown = 0
    CONFIDENCE_THRESHOLD = 0.85

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        current_word = ""

        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:

                landmarks = []
                for lm in hand_lms.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)

                landmarks = np.array(landmarks).reshape(1, -1)

                prediction = model.predict(landmarks, verbose=0)
                confidence = np.max(prediction)
                class_id = np.argmax(prediction)

                if confidence >= CONFIDENCE_THRESHOLD:
                    current_word = class_names[class_id]

                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS
                )

        # -------- SENTENCE LOGIC WITH CLEAR --------
        if current_word and cooldown == 0 and current_word != previous_word:
            if current_word.lower() == "clear":
                sentence_words.clear()
                previous_word = ""
            else:
                sentence_words.append(current_word)
                previous_word = current_word
            cooldown = 15

        if cooldown > 0:
            cooldown -= 1

        current_sentence = " ".join(sentence_words)

        # -------- DISPLAY CAMERA --------
        cv2.putText(
            frame,
            f"Sentence: {current_sentence}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hand Gesture to Text", frame)

        # -------- STOP CAMERA --------
        if cv2.waitKey(1) & 0xFF == ord("q"):
            camera_running = False
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/userdetectgesture", methods=["GET", "POST"])
def userdetectgesture():
    global camera_running

    if request.method == "POST":
        if not camera_running:
            camera_running = True
            threading.Thread(target=camera_loop, daemon=True).start()

    return render_template("userdetectgesture.html")


@app.route("/get_text")
def get_text():
    return jsonify({"text": current_sentence})


# ---------------- AUTO OPEN BROWSER ----------------
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


# ---------------- RUN ----------------
if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True, use_reloader=False)
