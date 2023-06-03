# ERA-Session-5
# PyTorch CNN Classifier

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) classifier. The CNN model is designed to classify images from the MNIST dataset into one of the ten digits (0-9).

## Table of Contents

- [Project Structure](#Project Structure)
- [Net Class](#Net Class)
- [Model Class](#Model Class)
- [Utils](#Utils)

## Project Structure

The Project has been made module by moving the components into 2 files:
1. model.py - This is the file where the actual neural network model is defined and run
2. utils.py - All the utility functions have been moved into this file
3. Session_5.ipynb - This is the colab notebook which imports the model.py file and the utils.py file and runs the model

## Net Class in model.py

The `Net` class defines the structure of the CNN model. It consists of four convolutional layers followed by two fully connected layers. The model architecture is as follows:

- Convolutional layer 1: 1 input channel, 32 output channels, kernel size of 3x3
- Convolutional layer 2: 32 input channels, 64 output channels, kernel size of 3x3
- Convolutional layer 3: 64 input channels, 128 output channels, kernel size of 3x3
- Convolutional layer 4: 128 input channels, 256 output channels, kernel size of 3x3
- Fully connected layer 1: 4096 input features, 50 output features
- Fully connected layer 2: 50 input features, 10 output features (corresponding to the 10 classes)


## Model class in model.py

The `Model` class stores the actual running of the CNN model along with the results, which can be later utilized to plot the results of the network.

- The `Model` class is responsible for training and evaluating the CNN model. It takes the `train_loader` and `test_loader` as the positional arguments, `learning_rate` and `momentum` as optional args. It initializes the model from the Net Class, defines the optimizer based on the args sent, and also the scheduler which will be used in the `train` and `test` methods. 

- The training and testing processes are implemented in the `train` and `test` methods, respectively. In the `train` method we perform the following operations:
    1. Set the model to training mode as the first step
    2. Then we iterate over the batches in the training data loader
    3. Moves the data and target tensors to the device (CPU or GPU) based on the availability of CUDA.
    4. Clears the gradients of all optimized parameters.
    5. Performs a forward pass through the model to obtain the predicted output probabilities for the input data.
    6. Calculates the loss using the cross-entropy loss function. This function combines a softmax activation function and the negative log-likelihood loss.
    7. Updates the running sum of the training loss by adding the current batch loss.
    8. Backpropagates the gradients through the network, computing the gradients of the loss with respect to the model parameters.
    9. Updates the model parameters using the computed gradients and the specified optimization algorithm (in this case, SGD).
    10. Appends the training accuracy and training loss to the results dictionary for later analysis

- In the `test` method we perform the following operations:
    1. Set the model to evaluation mode as the first step
    2. We have to Temporarily disable gradient calculation and gradient updates. This is done to speed up the evaluation process and reduce memory consumption.
    3. Then we iterate over the batches in the testing data loader
    4. 
- The `run` method executes the training and testing for the specified number of epochs.

## Utils functions

- `get_train_transforms()`: Returns a composition of data transformations for the training dataset. The transformations include random center cropping, resizing, random rotation, converting to a tensor, and normalization.

- `get_test_transforms()`: Returns a composition of data transformations for the testing dataset. The transformations include converting to a tensor and normalization.

- `get_train_and_test_mnist_data()`: Downloads and returns the MNIST training and testing datasets using the specified data transformations obtained from get_train_transforms() and get_test_transforms().

- `get_train_and_test_mnist_dataloader(batch_size)`: Creates and returns the data loaders for the training and testing datasets. The function takes an optional argument batch_size to specify the batch size for the data loaders.

- `plot_train_data(train_loader)`: Plots a grid of 12 images from the training data loader. It retrieves the first batch of images and labels from the data loader and displays them using matplotlib.

- `plot_model_results(results)`: Plots the training and testing results of a model. The function takes a dictionary results containing the training and testing losses and accuracies. It creates a 2x2 grid of subplots and plots the training loss, training accuracy, testing loss, and testing accuracy on separate subplots using matplotlib.

