# Flower Image Classification with kNN

A learning project that classifies flower images using colour histograms and scikit-learn's k-Nearest Neighbours classifier. The five classes are daisy, dandelion, rose, sunflower, and tulip.

The image-loading and feature-extraction pipeline is written in Python using OpenCV and NumPy. The kNN classifier itself uses scikit-learn rather than a from-scratch implementation.

## How it works

1. Load images from a separate folder for each species, using the folder name as the label. Display loading progress with tqdm.
2. Resize each image to 150 x 150 pixels.
3. Split the images into approximately 80% training, 10% validation, and 10% testing. Both splits use stratification to preserve class proportions and `random_state=1`.
4. Extract a joint colour histogram across OpenCV's three BGR channels. With 6 bins per channel, each image becomes 6 x 6 x 6 = 216 features. Normalise each histogram to the range 0 to 1 and flatten it into one vector.
5. Try k values from 1 to 99, training on the training features and selecting the highest validation accuracy. If scores tie, select the first (smallest) k.
6. Fit a final classifier on the training features using the selected k. The current version uses distance-weighted voting and Euclidean distance.
7. Evaluate test predictions using accuracy, macro precision, macro recall, macro F1-score, and a confusion matrix.

Distance-weighted voting gives closer neighbours more influence. Macro metrics calculate a score for each species and average those scores equally.

## Setup and running

Use a Python environment with these packages installed:

```powershell
python -m pip install numpy opencv-python tqdm scikit-learn matplotlib
```

Arrange the dataset beside the script:

```text
knn-flower-images/
├── README.md
├── flower_knn_color_histograms.py
└── flowers/
    ├── daisy/
    ├── dandelion/
    ├── rose/
    ├── sunflower/
    └── tulip/
```

Place readable image files directly inside their corresponding species folders. The flower dataset used in the tutorial contains 4,317 images. If the dataset is not included in your checkout, obtain it from the course materials and extract it into this structure.

From the repository root, run:

```powershell
cd knn-flower-images
python flower_knn_color_histograms.py
```

Run from this folder because the script looks for `flowers` relative to the terminal's current directory. Use your selected Python environment for both installation and execution.

## Outputs

- Progress bars while loading each species.
- The selected k and its validation accuracy (`Optimal Accuracy`).
- The final model's test accuracy (`Optimal Model Accuracy`).
- Test accuracy, macro precision, macro recall, and macro F1-score as fractions from 0 to 1.
- A confusion matrix window: rows are actual species and columns are predicted species.

The current script labels all four final metric lines `Accuracy`; their order is accuracy, precision, recall, then F1-score. The matrix is displayed rather than automatically saved.

## Experiments and results

Recorded results from development runs using the 80/10/10 split:

| Setup | Voting            | Histogram bins/channel | Best `k` | Validation Accuracy | Test Accuracy |
| ----- | ----------------- | ---------------------: | -------: | ------------------: | ------------: |
| 1     | Uniform           |                      6 |       39 |                ~50% |        41.67% |
| 2     | Distance-weighted |                      6 |       39 |                ~52% |          ~44% |
| 3     | Not recorded      |                      8 |       31 |                ~49% |        42.82% |


Distance-weighted voting with 6 bins per channel achieved the highest recorded validation accuracy. Increasing the histogram size to 8 produced 512 features but did not improve validation performance in the reported experiment. The best k is specific to the data, split, features, and voting settings; it is not a universal value.

Validation and some test scores were rounded when printed. These are recorded experiment results, not guaranteed scores for every run. A fixed random seed reproduces the split only when the input ordering is the same; the loader uses `os.listdir()` without sorting, so file ordering can differ across systems.

The course reference run used a different 60/20/20 split and achieved 43.06% test accuracy with k=11. That result is not a controlled comparison with this project's split. Compare model settings using the same training and validation samples. Because test results were inspected across experiments, treat the reported test scores as exploratory rather than a completely independent final evaluation.

## Limitations and learning outcomes

Colour histograms describe colour distribution but discard spatial arrangement and flower shape. Similar colours, image backgrounds, and lighting differences can make species difficult to distinguish.

This project practises image preprocessing, feature extraction, stratified data splitting, selecting k using validation data, distance-weighted voting, and interpreting classification metrics. It demonstrates that adding more histogram bins does not automatically improve classification.

