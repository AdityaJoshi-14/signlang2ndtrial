import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------- CONFIG -------- #
DATA_PATH = os.path.join('data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30
MODEL_PATH = os.path.join('models', 'action.h5')
# ------------------------ #

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

print("🔍 Loading data...")
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            frame_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(frame_path):
                window.append(np.load(frame_path))
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"[⚠️] Skipping {action} seq {sequence} (only {len(window)}/30 frames)")

# Convert to numpy arrays
if len(labels) == 0:
    raise ValueError("❌ No training data found. Recheck your data collection.")

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"✅ Loaded {len(X)} sequences.")
unique, counts = np.unique(labels, return_counts=True)
print("🧾 Label distribution:", dict(zip(actions, counts)))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y)

# -------- MODEL -------- #
model = Sequential([
    Input(shape=(sequence_length, 1662)),
    LSTM(64, return_sequences=True, activation='relu'),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# -------- TRAIN -------- #
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# -------- SAVE -------- #
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved at: {MODEL_PATH}")

# -------- EVALUATE -------- #
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=actions))

# -------- CONFUSION MATRIX -------- #
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=actions, yticklabels=actions, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

