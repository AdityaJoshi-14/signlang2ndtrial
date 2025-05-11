import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------- CONFIG -------- #
action = 'hello'
DATA_PATH = os.path.join('data', action)
MODEL_PATH = os.path.join('models', f'{action}_model.keras')
no_sequences = 30
sequence_length = 30
# ------------------------ #

# -------- LOAD DATA -------- #
X, y = [], []
for seq in range(no_sequences):
    window = []
    for frame_num in range(sequence_length):
        path = os.path.join(DATA_PATH, str(seq), f"{frame_num}.npy")
        if os.path.exists(path):
            window.append(np.load(path))
    if len(window) == sequence_length:
        X.append(window)
        y.append(1)  # 1 = "hello"

# Generate synthetic negatives (all zeros)
X_neg = np.zeros_like(X)
y_neg = [0] * len(X_neg)

# Combine
X_all = np.array(X + list(X_neg))
y_all = np.array(y + y_neg)

# -------- SPLIT DATA -------- #
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all)

# -------- MODEL -------- #
model = Sequential([
    Input(shape=(sequence_length, 1662)),
    LSTM(64, return_sequences=True, activation='relu'),
    Dropout(0.3),
    LSTM(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)

# -------- TRAIN -------- #
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------- EVALUATE -------- #
print("\n✅ Training complete. Model saved at:", MODEL_PATH)
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Hello', 'Hello']))

# -------- CONFUSION MATRIX -------- #
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Not Hello', 'Hello'], yticklabels=['Not Hello', 'Hello'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

