# Machine Learning Projects

A collection of projects documenting my progress in machine learning,
including implementing algorithms, preparing data, and evaluating predictions.

## Running a Project

Each project folder contains its own README with setup instructions
and dependency installation commands or a requirements.txt file.

## Projects

### [Iris Classification with kNN](knn-iris/)

- Implements Euclidean distance, neighbour selection, and majority voting.
- Uses NumPy and pandas for calculations and data preparation.
- Uses scikit-learn for train/test splitting and accuracy evaluation.

### [Flower Image Classification with kNN](knn-flower-images/)

- Extracts colour histogram features from flower images using OpenCV.
- Selects k using validation accuracy and explores distance-weighted voting.
- Evaluates predictions using accuracy, precision, recall, F1-score,
  and a confusion matrix.

### [Flower Image Classification with MLPs](mlp-flower-images/)

- Classifies colour histogram features using scikit-learn and Keras MLPs.
- Compares hidden-layer architectures using validation accuracy.
- Uses early stopping for Keras and reports test classification metrics.

### [Flower Image Classification with a Keras CNN](cnn-flower-images/)

- Learns image features using three convolution and pooling blocks.
- Uses data augmentation, dropout, and early stopping.
- Reports classification metrics, training curves, a confusion matrix,
  and example predictions; recorded test accuracy is 69.21%.
