import numpy as np
import os
import argparse
import hashlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import seaborn as sns

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser()
parser.add_argument("--noise", action="store_true", help="Use noisy data")
parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
parser.add_argument("--batch", type=int, default=32, help="Batch size")
args = parser.parse_args()

# === PATH SETUP ===
base_path = "/home/ajinkya/AI_False_Target_Generator"
X_path = os.path.join(base_path, "ai_module/X_noisy_v2.npy" )
y_path = os.path.join(base_path, "ai_module/y_noisy_v2.npy" )

# === LOAD DATA ===
X = np.load(X_path)
y = np.load(y_path)
X = X[..., np.newaxis]  # (samples, time_steps, 1)

# === SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL ===
model = models.Sequential([
    layers.Conv1D(32, 5, activation='relu', input_shape=(X.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
log_dir = os.path.join(base_path, "logs")
os.makedirs(log_dir, exist_ok=True)
callbacks_list = [
    callbacks.TensorBoard(log_dir=log_dir),
    callbacks.ModelCheckpoint(
        filepath=os.path.join(base_path, "best_model.h5"),
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
]

# === TRAIN ===
model.fit(
    X_train, y_train,
    epochs=args.epochs,
    batch_size=args.batch,
    validation_split=0.1,
    callbacks=callbacks_list
)

# === EVALUATE ===
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# === CONFUSION MATRIX ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("conf_matrix.png")
plt.show()

# === SAVE FINAL MODEL ===
model_path = os.path.join(os.path.dirname(__file__), "radar_classifier.h5")
model.save(model_path)
print(f"✅ Model saved at: {model_path}")

# === OVERFITTING CHECK ===
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"📈 Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f}")

# === LEAK CHECK ===
hashes_train = set([hashlib.md5(x.tobytes()).hexdigest() for x in X_train])
hashes_test = set([hashlib.md5(x.tobytes()).hexdigest() for x in X_test])
leakage = hashes_train.intersection(hashes_test)
print(f"🕵️‍♂️ Leaked samples: {len(leakage)}")

# === SAMPLE VISUALIZATION ===
plt.plot(X_train[0].squeeze())
plt.title(f"Label: {int(y_train[0])}")
plt.savefig("s.png")
plt.show()

# === BAD PREDICTIONS VISUALIZATION ===
bad_idx = np.where(y_pred.reshape(-1) != y_test)[0]
if len(bad_idx):
    print(f"👀 Visualizing {min(3, len(bad_idx))} misclassified samples...")
    for i in bad_idx[:3]:
        plt.plot(X_test[i].squeeze())
        plt.title(f"❌ True: {int(y_test[i])} | Pred: {int(y_pred[i])}")
        plt.show()
else:
    print("🎯 No misclassifications detected.")
