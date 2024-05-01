# ML_Project46-DigitRecognitionMNISTsequence

### Digit Sequence Recognition Model

This repository contains the code for a Convolutional Neural Network (CNN) model trained to recognize sequences of digits using the MNIST dataset.

#### Importing Libraries:
Necessary libraries like random, os, numpy, scipy, tensorflow, h5py, keras, and matplotlib are imported.

#### Setting Random Seed:
random.seed(101) sets a random seed for reproducibility of the results.

#### Defining MNIST Image Dimensions:
Variables mnist_image_height and mnist_image_width are set to 28, representing the dimensions of each MNIST digit image.

#### Loading MNIST Data:
(X_train, y_train), (X_test, y_test) = mnist.load_data() loads the MNIST training and testing datasets.

X_train and X_test contain the image data, while y_train and y_test contain the corresponding labels.

#### Checking the Data:
The code prints the shapes of the training and testing datasets and displays a sample image with its label.

#### Building Synthetic Data:
The build_synth_data function generates synthetic training and testing data by concatenating multiple MNIST digit images horizontally.

It randomly picks up to 5 digits from the MNIST dataset, stitches them together, and adds blanks for shorter sequences.

This function creates a dataset of 60,000 training and 10,000 testing images with corresponding labels encoded as tuples.

#### Preprocessing Labels:
The convert_labels function converts the labels from single digits to one-hot encoded arrays for each digit position (5 in total).

This allows the model to predict the individual digit in each position of the sequence.

#### Preprocessing Images:
The prep_data_keras function reshapes the image data to fit the Keras model format, converts them to floats, and normalizes pixel values between 0 and 1.

#### Building the Model:
A Convolutional Neural Network (CNN) is built using Keras.
The architecture involves:
```
Convolutional layers with ReLU activations for feature extraction.
MaxPooling layers for reducing dimensionality.
Dropout layers to prevent overfitting.
A flattening layer to prepare data for classification.
Five Dense layers with softmax activation for predicting each digit position (11 classes including blank).
```

### Compiling and Training the Model:
The model is compiled with categorical cross-entropy loss, Adam optimizer, and accuracy metric.

It's trained for 12 epochs on the synthetic training data with validation on the synthetic testing data.

### Key Points:
This code demonstrates how to build a CNN for recognizing sequences of digits using the MNIST dataset.

It addresses the challenge of multi-digit recognition by creating synthetic data and using multiple output layers.

The code includes data preprocessing, model building, training, and evaluation steps.

### Key Features:
Built using Keras for efficient deep learning development.

Utilizes synthetic data generation to address multi-digit recognition challenges.

Employs a CNN architecture with multiple output layers for predicting each digit in the sequence.

Achieves high accuracy on the MNIST dataset.

### Instructions:

#### Install Requirements:
Ensure you have the necessary libraries installed: tensorflow, keras, numpy, matplotlib, etc. You can install them using pip install <package_name>.

#### Run the Code:
Execute the Python script (main.py) to train the model. This will:
```
Load the MNIST dataset.
Generate synthetic training and testing data.
Preprocess the data.
Build, compile, and train the CNN model.
```

### Output:

The training process will display the loss and accuracy metrics on each epoch. After training, the model weights will be saved as model.h5.

### Further Exploration:

You can modify the model architecture (number of layers, neurons, etc.) to experiment with different configurations.

Explore different synthetic data generation techniques to create more diverse training data.

Visualize the learned filters and activations within the CNN layers to gain insights into the model's behavior.

Feel free to contribute or raise any issues related to this project.
