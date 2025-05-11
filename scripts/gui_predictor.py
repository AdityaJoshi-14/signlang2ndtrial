import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import threading

# ----------------- CONFIG ----------------- #
MODEL_PATH = 'models/action.h5'
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30
threshold = 0.3  # Lowered for better sensitivity
# ------------------------------------------ #

# Load model
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                    ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                    ).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                  ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                  ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])

# GUI Application
class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detector")
        
        # Webcam feed display
        self.video_label = Label(self.root)
        self.video_label.pack(pady=10)
        
        # Prediction label
        self.prediction_label = Label(self.root, text="Prediction: ...", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)
        
        # Buttons
        self.start_button = tk.Button(self.root, text="Start", command=self.start_detection, font=("Helvetica", 12))
        self.start_button.pack(side=tk.LEFT, padx=20, pady=10)
        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_detection, font=("Helvetica", 12))
        self.stop_button.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # State variables
        self.running = False
        self.sequence = []
        self.pred_buffer = []  # For smoothing predictions
        
    def start_detection(self):
        if not self.running:
            self.running = True
            self.prediction_label.config(text="Prediction: Starting...")
            threading.Thread(target=self.run_detection, daemon=True).start()
    
    def stop_detection(self):
        self.running = False
        self.prediction_label.config(text="Prediction: Stopped")
    
    def run_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.prediction_label.config(text="Error: Webcam not accessible")
            self.running = False
            return
        
        # Set lower resolution for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.prediction_label.config(text="Error: Failed to read frame")
                    break
                
                # Make detections
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                
                # Predict
                keypoints = extract_keypoints(results)
                print(f"Real-time keypoints: {np.count_nonzero(keypoints)} non-zero values")  # Debug
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-sequence_length:]
                
                if len(self.sequence) == sequence_length:
                    res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    self.pred_buffer.append(res)
                    self.pred_buffer = self.pred_buffer[-3:]  # Smooth over last 3 predictions
                    avg_res = np.mean(self.pred_buffer, axis=0)
                    pred = actions[np.argmax(avg_res)]
                    conf = avg_res[np.argmax(avg_res)]
                    
                    if conf > threshold:
                        self.prediction_label.config(text=f"Prediction: {pred} ({conf:.2f})")
                    else:
                        self.prediction_label.config(text=f"Prediction: {pred} (Low confidence: {conf:.2f})")
                
                # Update webcam feed in Tkinter
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
                # Allow Tkinter to update
                self.root.update()
            
            cap.release()
        
        # Clear video feed when stopped
        self.video_label.configure(image='')
        self.prediction_label.config(text="Prediction: Stopped")

# Start GUI
if __name__ == "__main__":
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))  # Debug GPU usage
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
