# covid-ct-classification
Deep learning model for classifying COVID-19 from chest CT scan images using TensorFlow and Keras.

# COVID-19 Detection from Chest CT Scans

A deep learning project for classifying chest CT scan images as **COVID-19** or **Non-COVID** using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Overview

The goal of this project is to explore how deep learning can be used for automated analysis of chest CT images.

The model is trained on the **COVIDx CT dataset** and learns visual patterns from CT scans to distinguish between COVID-19 and non-COVID cases.

> This project is intended for learning and research purposes and is not designed for clinical diagnosis.

## Dataset

The project uses the **COVIDx CT dataset** available on Kaggle:

[COVIDx CT Dataset](https://www.kaggle.com/datasets/hgunraj/covidxct)

The dataset contains labeled chest CT images from clinical cases.

For this project, the original labels are converted into two classes:

* `0` — Non-COVID
* `1` — COVID-19

To make the project easier to run locally, a subset of the available data is used for training, validation, and testing. The notebook notes that the full dataset can be used when running the project with suitable GPU resources.

## Model

A custom CNN was implemented using TensorFlow/Keras.

The architecture includes:

* 3 convolutional blocks
* Batch normalization
* Max pooling
* A fully connected layer
* Dropout for regularization
* A sigmoid output layer for binary classification

The model is trained using:

* **Optimizer:** Adam
* **Learning rate:** `1e-4`
* **Loss:** Binary Cross-Entropy
* **Batch size:** `32`
* **Input size:** `224 × 224`
* **Early stopping:** enabled based on validation loss

## Image Preprocessing

Before being passed to the model, CT images are:

1. Loaded from disk
2. Decoded as RGB images
3. Resized to `224 × 224`
4. Converted to floating-point values
5. Normalized to the `[0, 1]` range

The training dataset is shuffled and TensorFlow's `tf.data` pipeline is used for batching and prefetching.

## Training

The model is configured to train for up to 50 epochs, with early stopping used to prevent unnecessary training once validation performance stops improving.

Training metrics are saved to:

```text
training_log.csv
```

The project also includes TensorBoard logging for monitoring the training process.

## Evaluation

Model performance is evaluated on a separate test set.

The notebook reports:

* Accuracy
* Classification report
* Confusion matrix
* Training and validation loss
* Training and validation accuracy

A confusion matrix is also generated to visualize the model's predictions for the two classes.

## Project Structure

```text
covid-ct-classification/
│
├── data/
│   ├── 3A_images/
│   ├── train_COVIDx_CT-3A.txt
│   ├── val_COVIDx_CT-3A.txt
│   └── test_COVIDx_CT-3A.txt
│
├── code.ipynb
├── covid_ct_model.h5
├── training_log.csv
├── pixi.toml
├── pixi.lock
└── README.md
```

## Installation

The project uses [Pixi](https://pixi.sh/) for environment and dependency management.

The main dependencies include:

* Python 3.12
* TensorFlow
* Keras
* NumPy
* Pandas
* scikit-learn
* Matplotlib
* Seaborn
* Jupyter

After installing Pixi, the environment can be created from the project configuration.

## Running the Project

Clone the repository and open the notebook:

```bash
git clone https://github.com/your-username/covid-ct-classification.git
cd covid-ct-classification
pixi install
pixi run jupyter notebook
```

Then open:

```text
code.ipynb
```

Make sure the dataset is placed under the expected `data/` directory before running the notebook.

## Model Output

The trained model is saved as:

```text
covid_ct_model.h5
```

It can be loaded later with TensorFlow/Keras for further evaluation or inference.

## Notes

This project is a practical implementation of a CNN-based medical image classification workflow, covering the main steps from dataset preparation and image preprocessing to model training and evaluation.

The current version uses a limited subset of the dataset to keep training manageable on a local machine. Using the complete dataset and GPU acceleration would allow for further experimentation and potentially more robust results.

## Author

**Mohammad Shafiei**

This project was developed as a deep learning exercise in medical image classification using TensorFlow and Keras.
