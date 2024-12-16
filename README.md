# Brain Tumor Detection

This project uses deep learning to classify brain tumor types based on medical images. The model is built using TensorFlow and Keras, and it classifies brain tumors into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**.

## Project Structure

brain-tumor-detection/
│
├── data/
│   ├── Training/
│   └── Testing/
├── model/
│   └── brain_tumor_detection_model.h5  (Pre-trained model)
└──src/
    ├── train_model.py   
    ├── data_preparation.py
    ├── use_model.py
    ├── predict.py
    ├── README.md
    └── requirements.txt

## Script Overview

1. **`data_preparation.py`**:
   - Contains functions to prepare the image data for training and testing.
   - Defines image augmentation techniques such as rotation, brightness adjustment, and flipping to enhance the training data.

2. **`train_model.py`**:
   - Defines the model architecture using a Convolutional Neural Network (CNN).
   - Trains the model using the prepared data and saves the trained model as `brain_tumor_detection_model.h5`.

3. **`use_model.py`**:
   - Loads the trained model and evaluates its performance on the test data, displaying metrics like accuracy, confusion matrix, and classification report.

4. **`predict.py`**:
   - Accepts a user-provided image path, preprocesses the image, and predicts the tumor type using the trained model.

5. **`requirements.txt`**:
   - Lists all the required Python packages to run the project.

## Installation

To run this project, you will need to have Python 3.6+ installed. You can create a virtual environment and install the necessary dependencies by running:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

```

Dependencies

    tensorflow
    numpy
    scikit-learn

Alternatively, you can install the dependencies manually by running:

pip install tensorflow numpy scikit-learn

## Running the Project
1. **`Preparing and Training the Model`**

To train the model on your dataset:
```bash
python train_model.py
```
This will load the training data from the data/Training directory, train the model, and save it as model/brain_tumor_detection_model.h5.
2. **`Evaluating the Model`**

Once the model is trained or if you have a pre-trained model, you can evaluate its performance on the test data:
```bash
python use_model.py
```
This will load the model, evaluate it on the test data, and print the accuracy, loss, confusion matrix, and classification report.
3. **`Making Predictions`**

To make predictions on a new image:
```bash
python predict.py
```
You will be prompted to enter the path to the image. The script will process the image and print the predicted tumor type.
Data

Ensure that you have your training and testing images organized in directories. The directory structure should look like this:

data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
├── Testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/

The images should be labeled according to the tumor type and stored in the appropriate subdirectory.
## License

This project is licensed under the MIT License - see the LICENSE file for details.
