import cv2
import numpy as np
import os
from pathlib import Path 
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import Input
from keras.callbacks import EarlyStopping

def preprocess_images(image, img_size=256):
    """Downsize images"""
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)


def extract_color_histogram(dataset, hist_size=6):
    """Extract color histogram"""
    col_hist = []
    for img in dataset:
        img_hist = cv2.calcHist([img], [0, 1, 2], None, [hist_size, hist_size, hist_size], [0, 256, 0, 256, 0, 256])
        img_hist = cv2.normalize(img_hist, None, 0, 1, cv2.NORM_MINMAX).flatten()
        col_hist.append(img_hist)
    return col_hist

dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "flowers"

def load_dataset(dataset_path):
    """Load dataset: save each img and their label accordingly"""

    X, Y = [], []

    for species in os.listdir(dataset_path):
        species_path = os.path.join(dataset_path, species)
        for img in tqdm(os.listdir(species_path), desc = f"Processing: {species}"):
            img_path = os.path.join(species_path, img)
            processed_img = preprocess_images(img_path)
            X.append(processed_img)
            Y.append(species)
    return X, Y

X, Y = load_dataset(dataset_path)

# Split dataset 80% train, 10% validate, 10% test

x_train, x_temp, y_train, y_temp = train_test_split(
    X,
    Y, 
    train_size= 0.8,
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

print(f"Training Samples: {len(x_train)/len(X)*100:.1f}%")
print(f"Validation Samples: {len(x_val)/len(X)*100:.1f}%")
print(f"Test Samples: {len(x_test)/len(X)*100:.1f}%")

# Extract color histograms 

x_train_col = extract_color_histogram(x_train)
x_val_col = extract_color_histogram(x_val)
x_test_col = extract_color_histogram(x_test)

# Assign hidden layers & achitectures
# Based on 3 rules of thumb for heuristics

n_hidden_1 = 100
n_hidden_2 = 149
n_hidden_3 = 400

n_hidden_options = [
    (n_hidden_1,),
    (n_hidden_2,),
    (n_hidden_3,),
    (n_hidden_1, n_hidden_2),
    (n_hidden_2, n_hidden_3),
    (n_hidden_1, n_hidden_3),
    (n_hidden_1, n_hidden_2, n_hidden_3),
    (n_hidden_3, n_hidden_2, n_hidden_1),
    (n_hidden_2, n_hidden_3, n_hidden_1),
]

# Test each architecture and save their scores
val_scores = []

for architectuure in n_hidden_options:
    model = MLPClassifier(
        hidden_layer_sizes=architectuure, 
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=1,
    )
    results = model.fit(x_train_col, y_train)
    val_accuracy = model.score(x_val_col, y_val)
    val_scores.append(val_accuracy)

# Find use best score's architecture to retrain the model 
# Apply to test dataset

best_score = max(val_scores)
hidden_option_index = val_scores.index(best_score)
best_architecture = n_hidden_options[hidden_option_index]
print(f"The best architecture is: {best_architecture}")
print(f"Validation score of: {best_score:.2%}")

best_model = MLPClassifier(
    hidden_layer_sizes=(best_architecture),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=1,
)

best_model.fit(x_train_col, y_train)
y_pred = best_model.predict(x_test_col)

# Show results

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"recall: {recall*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=best_model.classes_,))   

# Plot confusion matrix

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=best_model.classes_,
    display_labels=best_model.classes_,
    cmap="Blues",
    xticks_rotation=45,
)

plt.title("Confusion Matrix - Scikit-Learn MLP", fontweight="bold")
plt.tight_layout()
plt.show()

# Convert labels to integers
label_encoder = LabelEncoder()
y_train_cat = to_categorical(label_encoder.fit_transform(y_train), 5)
y_val_cat = to_categorical(label_encoder.transform(y_val), 5)
y_test_cat = to_categorical(label_encoder.transform(y_test), 5)

# Convert color histograms from list to NumPy array containing 32-bit floating-point numbers:
x_train_col = np.asarray(x_train_col, dtype=np.float32)
x_val_col = np.asarray(x_val_col, dtype=np.float32)
x_test_col = np.asarray(x_test_col, dtype=np.float32)

# Early stop function no sizable adjustment has been made
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

best_keras_model = None
best_keras_score = -1
best_keras_architecture = None

for architecture in n_hidden_options:
    keras_mlp = Sequential()

    keras_mlp.add(
        Input(shape=(x_train_col.shape[1],))
    )
    for neuron in architecture:
        keras_mlp.add(
            Dense(neuron, activation="relu")
        )

    keras_mlp.add(
        Dense(5, activation="softmax")
    )

    keras_mlp.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = keras_mlp.fit(
        x_train_col,
        y_train_cat,
        validation_data=(x_val_col, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1,
    )

    _, val_accuracy = keras_mlp.evaluate(
    x_val_col, y_val_cat, verbose=0
    )

    if val_accuracy > best_keras_score:
        best_keras_score = val_accuracy
        best_keras_model = keras_mlp
        best_keras_architecture = architecture

print(f"Best Keras architecture: {best_keras_architecture}")
print(f"Validation accuracy: {best_keras_score:.2%}")

probabilities = best_keras_model.predict(x_test_col, verbose=0)
y_keras_pred = label_encoder.inverse_transform(
    np.argmax(probabilities, axis=1)
)

print(f"Keras test accuracy: {accuracy_score(y_test, y_keras_pred):.2%}")
print(classification_report(
    y_test,
    y_keras_pred,
    labels=label_encoder.classes_,
    target_names=label_encoder.classes_,
    zero_division=0,
))