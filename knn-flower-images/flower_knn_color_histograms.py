import numpy as np  
import os 
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def preprocess_images(image, img_size=150):
    """Downsize images"""
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_size, img_size))
    return np.array(img)

def extract_color_histogram (dataset, hist_size=6):
    """Extract color histogram"""
    col_hist = []
    for img in dataset:
        hist = cv2.calcHist(
            [img], [0, 1, 2], None,
            (hist_size, hist_size, hist_size),
            [0, 256, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, None, 0, 1, cv2.NORM_MINMAX)
        col_hist.append(hist.flatten())
    return np.array(col_hist)
flower_types = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def load_dataset(main_folder='flowers'):
    """Load dataset: save each img and their label accordingly"""

    X = [] 
    Y = []

    for flower in flower_types:
            type_path = os.path.join(main_folder, flower)
            for img in tqdm(os.listdir(type_path), desc=f"Loading: {flower}"):
                img_path = os.path.join(type_path, img)
                processed_img = preprocess_images(img_path)
                X.append(processed_img)
                Y.append(flower)
    return X, Y

X, Y = load_dataset()

# Split dataset 80% train, 10% validate, 10% test

X_train, X_temp, Y_train, Y_temp = train_test_split(
     X, 
     Y, 
     train_size=0.8,
     random_state=1,
     shuffle=True,
     stratify=Y
)

X_val, X_test, Y_val, Y_test = train_test_split(
     X_temp,
     Y_temp, 
     test_size=0.5,
     random_state=1,
     shuffle=True,
     stratify=Y_temp
)

X_train_col = extract_color_histogram(X_train)
X_val_col = extract_color_histogram(X_val)
X_test_col = extract_color_histogram(X_test)

# Find optimal model by adjusting k and finding highest 
# accuracy score on validation dataset

k = 1
last_k = 100

accuracy_scores = []

while k < last_k:
     
     model = KNeighborsClassifier(n_neighbors=k, weights='distance')
     model.fit(X_train_col, Y_train)
     accuracy = model.score(X_val_col, Y_val)
     accuracy_scores.append(accuracy)
     k+=1

optimal_accuracy = max(accuracy_scores)
optimal_k = accuracy_scores.index(optimal_accuracy) + 1
print(f"Optimal k : {optimal_k} \nOptimal Accuracy = {round(optimal_accuracy, 2)}")


# Retrain with optimal model

optimal_model = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
optimal_model.fit(X_train_col, Y_train)

# Test opitimal model on test dataset

test_accuracy = optimal_model.score(X_test_col, Y_test)
print(f"Optimal Model Accuracy: {round(test_accuracy, 2)}")

# Display performance results 

Y_pred = optimal_model.predict(X_test_col)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='macro', zero_division=0)
recall = recall_score(Y_test, Y_pred, average='macro', zero_division=0)
f1 = f1_score(Y_test, Y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy}")
print(f"Accuracy: {precision}")
print(f"Accuracy: {recall}")
print(f"Accuracy: {f1}")

# Plot confusion matrix

matrix_values = confusion_matrix(Y_test, Y_pred, labels=flower_types)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix_values,
    display_labels=optimal_model.classes_
)
display.plot()
plt.show()