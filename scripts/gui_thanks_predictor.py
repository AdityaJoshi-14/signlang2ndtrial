import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
import threading

# -------- CONFIG -------- #
MODEL_PATH = 'models/thanks_model.keras'
SEQUENCE_LENGTH = 30
THRESHOLD = 0.7
# ------------------------ #

# Load model
model = load_model(MODEL_PATH)
print("✅ Loaded model: thanks_model.keras")

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# GUI Application
class ThanksPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thanks Sign Detector")
        self.running = False
        self.sequence = []
        self.predictions = deque(maxlen=5)  # Smooth over last 5 predictions

        # GUI elements
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.prediction_label = tk.Label(root, text="Prediction: ...", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        self.start_btn = tk.Button(root, text="Start Detection", command=self.start_detection)
        self.start_btn.pack(side=tk.LEFT, padx=20, pady=10)

        self.stop_btn = tk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.stop_btn.pack(side=tk.RIGHT, padx=20, pady=10)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.prediction_label.config(text="Prediction: Stopped")
        self.video_label.config(image='')
        if hasattr(self, 'cap'):
            self.cap.release()

    def run_detection(self):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to grab frame")
                    break

                # Process frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Predict
                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-SEQUENCE_LENGTH:]

                if len(self.sequence) == SEQUENCE_LENGTH:
                    res = model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0][0]
                    self.predictions.append(res)
                    avg_pred = np.mean(self.predictions)

                    label = "Thanks" if avg_pred > THRESHOLD else "Not Thanks"
                    confidence = avg_pred if avg_pred > THRESHOLD else 1 - avg_pred
                    self.prediction_label.config(text=f"Prediction: {label} ({confidence:.2f})")

                # Update video feed
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                self.root.update()

        self.stop_detection()

if __name__ == "__main__":
    root = tk.Tk()
    app = ThanksPredictorApp(root)
    root.mainloop()
