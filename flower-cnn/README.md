# Flower Image Classification with a Keras CNN

Classify flower images as **daisy, dandelion, rose, sunflower, or tulip** using a convolutional neural network (CNN). Unlike the colour-histogram kNN and MLP projects, this model learns features directly from image pixels, including local patterns and spatial relationships.

This README covers [`flower-cnn_keras.py`](flower-cnn_keras.py), developed while following the [CSCI218 Week 5 CNN notebook](https://github.com/quangvnai/csci218/blob/main/Week5/week5-task2.ipynb). The architecture and dataset split described below belong to this implementation.

## Dataset

The script uses the shared dataset at `datasets/flowers`, with folder names providing the labels:

```text
machine-learning-projects/
|-- datasets/
|   `-- flowers/
|       |-- daisy/
|       |-- dandelion/
|       |-- rose/
|       |-- sunflower/
|       `-- tulip/
`-- flower-cnn/
    |-- flower-cnn_keras.py
    `-- README.md
```

The recorded run used **4,317 images**, split into **3,453 training**, **432 validation**, and **432 test** images. Both splits are stratified and use `random_state=1`, giving approximately 80/10/10 proportions.

The dataset path is resolved relative to the script, so it does not depend on the terminal's current directory. Place readable images directly inside each species folder; the loader does not skip unrelated files, nested folders, or unreadable images.

## Setup and execution

From the repository root, with your Python virtual environment activated:

```powershell
python -m pip install numpy opencv-python tqdm scikit-learn matplotlib tensorflow keras
python -u .\flower-cnn\flower-cnn_keras.py
```

Use the same interpreter for installation and execution. The development environment used Python 3.12. Training ran on the CPU; installing TensorFlow does not automatically enable every GPU.

The script prints loading progress, epoch metrics, and test results. It then displays training curves and a confusion matrix. Close those windows to continue to the correctly classified examples, then close that window to view the incorrectly classified examples. Models, plots, and reports are not saved automatically.

## Workflow

1. Load images with OpenCV and resize them to **256 x 256 pixels**.
2. Convert images into a single `float32` NumPy array and divide by 255 to normalise pixel values to 0-1. Images retain OpenCV's BGR channel order; display examples are converted to RGB.
3. Create stratified training, validation, and test splits.
4. Fit `LabelEncoder` on training labels and reuse its mapping for validation, testing, reports, and plots. Convert labels to five-class one-hot vectors.
5. Apply random horizontal flips, rotations (`RandomRotation(0.05)`), and zoom (`RandomZoom(0.1)`) during training.
6. Train the CNN with early stopping on validation loss and restore the best weights.
7. Evaluate the held-out test images, convert predicted probabilities to class indices with `argmax`, and report classification metrics.

## Model architecture

All convolution layers use 3 x 3 kernels, ReLU activation, and `padding="same"`. Each pooling layer uses a 2 x 2 window.

| Layer | Output shape per image |
|---|---|
| Training augmentation | 256 x 256 x 3 |
| Conv2D, 32 filters | 256 x 256 x 32 |
| MaxPooling2D | 128 x 128 x 32 |
| Conv2D, 64 filters | 128 x 128 x 64 |
| MaxPooling2D | 64 x 64 x 64 |
| Conv2D, 128 filters | 64 x 64 x 128 |
| MaxPooling2D | 32 x 32 x 128 |
| Flatten | 131,072 |
| Dense, ReLU | 32 |
| Dropout, rate 0.5 | 32 |
| Dense, softmax | 5 class probabilities |

`Flatten` retains the positions of features in the final maps. Its following dense layer contains about 4.2 million weights, so regularisation remains important. Dropout is active only during training.

## Training settings

| Setting | Value |
|---|---|
| Optimizer | Adam, default learning rate 0.001 |
| Loss | Categorical cross-entropy |
| Training metric | Accuracy |
| Maximum epochs | 30 |
| Batch size | 32 |
| Early-stopping monitor | Validation loss |
| Patience | 5 epochs without improvement |
| Restore best weights | Yes |

Each full epoch contains 108 training batches. Early stopping may finish sooner than 30 epochs, but it does not stop while validation loss continues improving frequently enough. Test metrics use the restored model, so they need not correspond to the final displayed epoch.

Training accuracy can be lower than validation accuracy because augmentation and dropout make training harder. Validation runs without those transformations. This difference alone does not demonstrate underfitting or a bug.

## Recorded results

These values were recorded from the supplied terminal output and plots, rather than a new benchmark run for this README.

| Test metric | Result |
|---|---:|
| Loss | 0.8711 |
| Accuracy | 69.21% |
| Macro precision | 0.6949 |
| Macro recall | 0.6871 |
| Macro F1 | 0.6836 |
| Weighted precision | 0.6962 |
| Weighted recall | 0.6921 |
| Weighted F1 | 0.6866 |
| Correct predictions | 299 / 432 |
| Average inference time | 3.856 ms/image |

Macro metrics give each class equal weight; weighted metrics account for class support. Inference time is the elapsed time of `model.predict` over the test set, using batches of 32, divided by the number of images. It includes prediction-call overhead and is not a standalone single-image latency benchmark or a hardware-independent result.

### Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Daisy | 0.6949 | 0.5325 | 0.6029 | 77 |
| Dandelion | 0.6692 | 0.8476 | 0.7479 | 105 |
| Rose | 0.6575 | 0.6076 | 0.6316 | 79 |
| Sunflower | 0.6932 | 0.8356 | 0.7578 | 73 |
| Tulip | 0.7595 | 0.6122 | 0.6780 | 98 |

### Confusion matrix

Rows are actual labels; columns are predicted labels. Diagonal entries are correct predictions.

| Actual / Predicted | Daisy | Dandelion | Rose | Sunflower | Tulip |
|---|---:|---:|---:|---:|---:|
| Daisy | 41 | 26 | 4 | 4 | 2 |
| Dandelion | 5 | 89 | 0 | 8 | 3 |
| Rose | 9 | 8 | 48 | 3 | 11 |
| Sunflower | 2 | 5 | 2 | 61 | 3 |
| Tulip | 2 | 5 | 19 | 12 | 60 |

Sunflowers achieved the highest F1 score, while daisies had the lowest recall. The largest confusion was **26 daisies predicted as dandelions**, followed by **19 tulips predicted as roses**. Displayed mistakes included small flowers, busy backgrounds, and mixed bouquets, which may make classification harder.

### Development experiments

| Configuration | Recorded test accuracy |
|---|---:|
| Two convolution blocks, without dropout or early stopping | 65.74% |
| Two convolution blocks, dropout 0.5 and early stopping | 67.82% |
| Three convolution blocks, dropout 0.5 and early stopping | 69.21% |

These are individual development runs, not controlled evidence that each change caused the improvement. The earlier model's training accuracy rose to about 87% while validation accuracy stayed around 75%, suggesting overfitting. In the latest plotted run, validation loss generally decreased and validation accuracy reached about 74% by the end.

## Limitations and further experiments

- Test results were inspected across development runs, so these are exploratory results rather than a completely independent final evaluation. Choose further settings using validation data.
- Keras training and augmentation are not seeded. File loading uses unsorted `os.listdir`, so the same split seed does not guarantee identical samples across systems. Results can also vary with dependency versions.
- All images are held in memory. The normalised image array alone occupies about 3.2 GiB, before split copies and training memory. CPU training at 256 x 256 can be slow.
- Try a higher epoch limit with early stopping if validation loss is still improving. Compare smaller image sizes, dropout rates, or learning rates as separate experiments.
- The CNN preserves more spatial information than histogram features, but comparisons with other projects require verifying identical sample splits and evaluation conditions.

This project practises image preprocessing, learned feature extraction, data augmentation, regularisation, early stopping, and interpreting class-specific errors.
