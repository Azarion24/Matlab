# Star Classification Project
vf
This project focuses on classifying stars based on features such as temperature, luminosity, radius, absolute magnitude, spectral class, and star color. The goal is to create and compare different classification models, including Neural Networks, Random Forest, SVM, and KNN, to predict the star type based on the given features.

## Contents

- `star_classification.csv` - Dataset containing information about stars (features like temperature, luminosity, radius, absolute magnitude, spectral class, and star color).
- `star_classification.m` - MATLAB script that performs data loading, feature preparation, model training, and performance evaluation.

## Code Description

### 1. Data Loading

The dataset is loaded from the `star_classification.csv` file, and initial preprocessing is performed:
- Missing data is removed.
- Numeric features are extracted and categorical labels are converted.
- Feature normalization is applied.

### 2. PCA (Principal Component Analysis)

PCA is performed to reduce the dimensionality of the data, which allows for visualization in a 2D space. The PCA results are displayed on a scatter plot, with different star types represented by different colors.

### 3. Data Splitting

The dataset is split into a training set (70%) and a test set (30%) using `cvpartition`. Models are trained on the training data and evaluated on the test data.

### 4. Model Training and Evaluation

- **Neural Network**: A fully connected neural network with ReLU activation function.
- **Random Forest**: An ensemble method using 100 decision trees.
- **SVM (Support Vector Machine)**: A linear kernel SVM model.
- **KNN (K-Nearest Neighbors)**: A KNN classifier with 5 nearest neighbors.

The accuracy of each model is calculated on the test set.

### 5. Confusion Matrix

After training the models, confusion matrices are displayed for each model to assess their performance.

### 6. Feature Importance

The feature importance for the Random Forest model is computed and displayed in a bar chart to highlight the significance of each feature.

## Requirements

To run this project, you'll need the following tools:

- MATLAB (preferably the latest version).
- Required MATLAB toolboxes:
  - Statistics and Machine Learning Toolbox
  - Deep Learning Toolbox (for Neural Networks)

### Installation Instructions

1. **Install MATLAB**: Download MATLAB from the [official MathWorks website](https://www.mathworks.com/products/matlab.html).

2. **Install the required toolboxes**:
   - Statistics and Machine Learning Toolbox
   - Deep Learning Toolbox

3. **Load the dataset**: Ensure that the `star_classification.csv` file is in the same directory as the MATLAB script (`star_classification.m`).

4. **Run the script**: After setting up the environment, execute the MATLAB script:
   ```matlab
   star_classification.m
