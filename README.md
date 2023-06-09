# Correlation Model Building

This code represents a class called `CorrelationModelBuilding` that automates the process of training and saving neural network models with a specific architecture. The class utilizes correlation analysis and data preprocessing techniques to select relevant columns from a given dataset and split it into training and testing sets. The trained models are then saved along with evaluation metrics such as ROC curves and confusion matrices.

## Prerequisites

Make sure you have the following libraries installed:
```python

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics
import os
from datetime import datetime
```
## Usage

1. Instantiate the `CorrelationModelBuilding` class:
```python
model = CorrelationModelBuilding(
    data, threshold, test_size, train_test_rand_seed, f1_lim, columns_to_drop, target_name, what_to_detect
)
```
- data (string): Path to the input CSV file containing the dataset.
- threshold (float): The correlation threshold value between 0 and 1. Columns with correlation values above this threshold will be considered.
- test_size (float): The proportion of the dataset to include in the test split. Should be between 0 and 1.
- train_test_rand_seed (int): Random seed for reproducibility when splitting the dataset.
- f1_lim (float): The minimum F1 score required to save a trained model.
- columns_to_drop (list of strings): Names of columns to drop from the dataset.
- target_name (string): Name of the target column.
- what_to_detect (string): Description of what the model is predicting.

## Preprocess the data:
This code puts splitted and processed data into variables.
```python
X_train, X_test, y_train, y_test = model.preprocess_data()
```

Train and save models:
Results




## 
The trained models, along with evaluation metrics, will be saved in separate folders with the following structure:

model_time_<current_time>__f1_<f1_score>/
  - model_ep_<epochs>_bt_<batch_size>_f1_<f1_score>_threshold_<correlation_threshold>_columns_<num_columns>_random_state_<random_seed>.h5
  - X_train.csv
  - X_test.csv
  - y_train.csv
  - y_test.csv
  - ROC_Curve.png
  - Confusion_Matrix.png
  ### Description 
    
    
- model_time_<current_time>__f1_<f1_score>: 
  - The folder name containing the trained model and related files.
- model_ep_<epochs>_bt_<batch_size>_f1_<f1_score>_threshold_<correlation_threshold>_columns_<num_columns>_random_state_<random_seed>.h5:
  - The saved model file in HDF5 format.
- X_train.csv, X_test.csv, y_train.csv, y_test.csv: 
  - CSV files containing the training and testing datasets.
- ROC_Curve.png:
  - ROC curve plot showing the model's performance.
- Confusion_Matrix.png:
  - Confusion matrix plot representing the model's predictions.
