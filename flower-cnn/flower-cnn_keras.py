import numpy as np
import cv2
from pathlib import Path
import os 
from timeit import default_timer as timer
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import RandomFlip, RandomRotation, RandomZoom, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping

def preprocess_image(image, img_size=256):
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)

dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "flowers"

def load_images(dataset_path):
    X, Y = [], []

    for species in os.listdir(dataset_path):
        species_path = os.path.join(dataset_path, species)
        for img in tqdm(os.listdir(species_path), desc=f"Loading: {species}"):
            processed_img = preprocess_image(os.path.join(species_path, img))
            X.append(processed_img)
            Y.append(species)
    return np.array(X, dtype=np.float32) / 255.0, np.array(Y)

def show_images(y_true, y_pred, images, correct=True, num_images=5):
    count = 0
    plt.figure(figsize=(15, 3))
    for i in range(len(y_true)):
        if (y_true[i] == y_pred[i]) == correct:

            count += 1

            plt.subplot(1, num_images, count)
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.title(
                f"Actual: {LABELS[y_true[i]]}\n"
                f"Predicted: {LABELS[y_pred[i]]}"
            )
            plt.axis("off")
            if count == num_images:
                break
    plt.show()

# Load dataset

X, Y = load_images(dataset_path)
print(f"Total samples loaded: {len(X)}")

# Split Dataset 80% / 10% / 10%

x_train, x_temp, y_train, y_temp = train_test_split(
    X, 
    Y,
    train_size=0.8,
    random_state=1,
    stratify=Y
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp,
    y_temp,
    test_size=0.5,
    random_state=1,
    stratify=y_temp
)

# Encode labels
# fit learns the class names and stores them in encode_label
# transform converts labels using that learned mapping
encode_label = LabelEncoder()
y_train_num = encode_label.fit_transform(y_train)
LABELS = encode_label.classes_
y_val_num = encode_label.transform(y_val)
y_test_num = encode_label.transform(y_test)

# Each numerical label create array of 5 classes symbolize 
# which species it is and is not
y_train_cat = to_categorical(y_train_num)
y_val_cat = to_categorical(y_val_num)
y_test_cat = to_categorical(y_test_num)

# Data Augmentation

augmented_data = Sequential([
    RandomFlip("horizontal"), 
    RandomRotation(0.05),
    RandomZoom(0.1),
])

# Setup CNN Architecture 

model = Sequential([
    augmented_data,
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    MaxPooling2D((2,2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(5, activation="softmax")
])

# Setup Learning Config

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Train CNN Model

history = model.fit(
    x_train,
    y_train_cat,
    validation_data=(x_val, y_val_cat),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
test_loss, test_accuracy = model.evaluate(
    x_test, 
    y_test_cat,
    verbose=0
    )
print(f"Test loss: {test_loss:.4f}")

# STEP 9. Plot Loss and Accuracy curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(history.history['accuracy'], label='Train Acc', lw=2)
ax1.plot(history.history['val_accuracy'], label='Val Acc', lw=2, linestyle='--')
ax1.set_title('CNN Accuracy vs. Epochs', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Train Loss', lw=2)
ax2.plot(history.history['val_loss'], label='Val Loss', lw=2, linestyle='--')
ax2.set_title('CNN Loss vs. Epochs', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# STEP 10. Benchmark inference latency
start = timer()
raw_predictions = model.predict(x_test, batch_size=32, verbose=0)
end = timer()

latency = (end - start) / len(x_test)
print(f"Inference latency: {latency:.6f} s/image ({latency * 1000:.3f} ms/image)")

# STEP 11. Report classification metrics
y_pred = np.argmax(raw_predictions, axis=1)
class_ids = np.arange(len(LABELS))
accuracy = accuracy_score(y_test_num, y_pred)
precision = precision_score(y_test_num, y_pred, labels=class_ids, average='macro', zero_division=0)
recall = recall_score(y_test_num, y_pred, labels=class_ids, average='macro', zero_division=0)
f1 = f1_score(y_test_num, y_pred, labels=class_ids, average='macro', zero_division=0)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro)   : {recall:.4f}")
print(f"F1-score (macro) : {f1:.4f}\n")
print(classification_report(
    y_test_num, y_pred, labels=class_ids, target_names=LABELS,
    digits=4, zero_division=0
))

# STEP 12. Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test_num, y_pred, labels=class_ids, display_labels=LABELS,
    cmap='Blues', xticks_rotation=45
)
plt.title('Confusion Matrix - Keras CNN')
plt.tight_layout()
plt.show()

# STEP 13. Show correctly and incorrectly classified images
show_images(y_test_num, y_pred, x_test, correct=True, num_images=5)
show_images(y_test_num, y_pred, x_test, correct=False, num_images=5)
     
