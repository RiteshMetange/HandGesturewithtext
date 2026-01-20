import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# ---------------- SETTINGS ----------------
DATASET_DIR = "dataset"
MODEL_DIR = "mp_hand_gesture"
GESTURE_NAMES_FILE = "gesture.names"
EPOCHS = 30
BATCH_SIZE = 16

# ---------------- LOAD DATA ----------------
X = []
y = []

gesture_names = []

for file in os.listdir(DATASET_DIR):
    if file.endswith(".csv"):
        gesture = file.replace(".csv", "")
        gesture_names.append(gesture)

        file_path = os.path.join(DATASET_DIR, file)
        data = np.loadtxt(file_path, delimiter=",")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        for row in data:
            X.append(row)
            y.append(gesture)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
print("Gestures:", gesture_names)

# ---------------- LABEL ENCODING ----------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# ---------------- SAVE gesture.names ----------------
with open(GESTURE_NAMES_FILE, "w") as f:
    for name in label_encoder.classes_:
        f.write(name + "\n")

print("gesture.names created")

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# ---------------- SAVE MODEL ----------------
model.save(MODEL_DIR)
print("Model saved to", MODEL_DIR)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
