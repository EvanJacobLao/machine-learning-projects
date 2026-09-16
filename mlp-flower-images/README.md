# Flower Classification with Scikit-Learn and Keras MLPs

Classify flower images as **daisy, dandelion, rose, sunflower, or tulip** using color histograms and multilayer perceptrons (MLPs) implemented with scikit-learn and Keras. Both implementations share the same features and dataset split.

This README covers [`flower_mlp_sklearn_keras.py`](flower_mlp_sklearn_keras.py).

## Dataset

The script uses the shared dataset in the repository's `datasets/flowers` directory. Folder names provide the class labels.

```text
machine-learning-projects/
├── datasets/
│   └── flowers/
│       ├── daisy/
│       ├── dandelion/
│       ├── rose/
│       ├── sunflower/
│       └── tulip/
└── mlp-flower-images/
    ├── flower_mlp_sklearn_keras.py
    └── README.md
```

Place valid image files directly inside each species folder. The current loader expects only species directories inside `flowers` and does not skip unreadable images or nested folders. The dataset path is resolved relative to the script.

## Setup and execution

From the repository root, with your Python virtual environment activated:

```powershell
python -m pip install numpy opencv-python tqdm scikit-learn matplotlib tensorflow keras
python -u .\mlp-flower-images\flower_mlp_sklearn_keras.py
```

The script runs scikit-learn first, prints its metrics and classification report, and opens its confusion-matrix plot. Close that plot window to continue to Keras training. Keras then prints epoch progress, the selected architecture, validation accuracy, test accuracy, and a classification report.

The current script does not plot Keras training curves or a Keras confusion matrix, and does not save models or plots automatically.

If using VS Code Code Runner, clear any text selection before running, or enable `code-runner.ignoreSelection`, to execute the full file.

## Workflow

1. Read images using OpenCV and resize them to **256 × 256 pixels**.
2. Split images into **80% training, 10% validation, and 10% test**, preserving class proportions with stratification and using `random_state=1`.
3. Compute a joint histogram over the three BGR color channels with **6 bins per channel**, producing **216 features per image**.
4. Min-max normalize each histogram to the range 0–1 and flatten it.
5. Train nine scikit-learn MLP architectures and compare validation accuracy. Retrain the selected architecture with the final configured settings and evaluate it on the test set.
6. Convert histogram features to `float32` NumPy arrays and encode flower labels as one-hot vectors for Keras.
7. Train the same nine architectures in Keras with early stopping. Keep the trained model with the highest validation accuracy and evaluate it on the test set.

The candidate hidden-layer architectures are:

```python
(100,), (149,), (400,),
(100, 149), (149, 400), (100, 400),
(100, 149, 400), (400, 149, 100), (149, 400, 100)
```

Each tuple describes the number of neurons in successive hidden layers. For example, `(149,)` means one hidden layer with 149 neurons.

## Scikit-learn model settings

| Setting | Architecture comparison | Final model |
|---|---|---|
| Hidden layers | Nine candidates above | Selected architecture |
| Activation | ReLU | ReLU |
| Optimizer | Adam | Adam |
| Maximum epochs | 500 | 500 |
| L2 regularization (`alpha`) | Default (`0.0001`) | `0.001` |
| Initial learning rate | Default (`0.001`) | `0.001` |
| Random state | 1 | 1 |

**Configuration note:** the final model uses the selected architecture, but changes `alpha` from `0.0001` to `0.001`. The printed validation score therefore belongs to the search configuration, not the final retrained configuration.

## Keras model settings

Each candidate is a `Sequential` model with a 216-feature input, ReLU hidden layers, and a five-neuron softmax output layer.

| Setting | Value |
|---|---|
| Hidden layers | Same nine candidates as scikit-learn |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Training metric | Accuracy |
| Maximum epochs | 100 per candidate |
| Batch size | 32 |
| Early-stopping monitor | Validation loss |
| Patience | 10 epochs without improvement |
| Restore best weights | Yes, from the lowest validation loss |
| Architecture selection | Highest validation accuracy after weight restoration |

`LabelEncoder` is fitted on training labels, then used to encode validation and test labels. `to_categorical` converts the encoded labels into five-class one-hot vectors. Predictions are converted from class probabilities to indices with `argmax`, then back to flower names using the same encoder for the classification report.

The selected Keras model is retained directly rather than retrained after selection. No Keras random seed is currently set, so repeated runs can select different architectures and produce different scores. The two implementations share data, but differ in training settings; this is not a controlled comparison of libraries alone.

## Recorded scikit-learn results

The following results come from the earlier scikit-learn run with final `max_iter=1000` and `alpha=0.001`, rather than a new run of the current combined script. No Keras benchmark results have been recorded here yet. The comparison selected `(149,)` with **67.59% validation accuracy**. The final model achieved:

| Test metric | Result |
|---|---:|
| Accuracy | 57.87% |
| Macro precision | 58.35% |
| Macro recall | 57.06% |
| Macro F1 | 57.21% |
| Test images | 432 |
| Correct predictions | 250 |

Macro metrics give each class equal weight.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Daisy | 0.53 | 0.58 | 0.56 | 77 |
| Dandelion | 0.57 | 0.68 | 0.62 | 105 |
| Rose | 0.52 | 0.37 | 0.43 | 79 |
| Sunflower | 0.72 | 0.60 | 0.66 | 73 |
| Tulip | 0.58 | 0.62 | 0.60 | 98 |

### Confusion matrix

Rows are actual labels; columns are predicted labels. Diagonal entries are correct predictions.

| Actual / Predicted | Daisy | Dandelion | Rose | Sunflower | Tulip |
|---|---:|---:|---:|---:|---:|
| Daisy | 45 | 22 | 2 | 3 | 5 |
| Dandelion | 14 | 71 | 5 | 9 | 6 |
| Rose | 9 | 11 | 29 | 2 | 28 |
| Sunflower | 9 | 11 | 4 | 44 | 5 |
| Tulip | 8 | 10 | 16 | 3 | 61 |

Sunflowers achieved the highest F1 score, while roses were the most difficult class. The largest individual confusion was roses predicted as tulips (28 images). Always predicting the most common test class, dandelion, would achieve 24.31% accuracy.

## Limitations and further experiments

- Color histograms capture color distribution but discard shape and spatial arrangement, which can make similarly colored flowers difficult to distinguish.
- A `ConvergenceWarning` means a model reached its iteration limit before meeting the convergence criterion. Predictions remain available; more epochs do not guarantee higher validation accuracy.
- Compare histogram bin counts, hidden-layer sizes, regularization, and learning rates using validation data or cross-validation. Keep test results out of model selection.
- Consider feature standardization fitted only on training data, or features that capture texture and shape.
- The loader holds all resized images in memory. Memory use grows with dataset size and image dimensions.
- Exact results can vary with dataset contents, file enumeration order, and library versions, even with a fixed random seed. Dependencies are not pinned for this script.
