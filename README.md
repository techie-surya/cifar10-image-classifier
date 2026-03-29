# A Beginner’s Guide to Image Classification using CNN (CIFAR-10)

## Objective

The goal of this project is to build a simple image classification model using a Convolutional Neural Network (CNN) that can identify objects like cats, dogs, cars, and more from images.

---

## What is Image Classification?

Image classification is a process where a computer learns to recognize and label images.

For example:

* Input → Image of a dog 
* Output → "Dog"

---

## What is CIFAR-10?

CIFAR-10 is a popular dataset used for learning image classification.

It contains:

* 60,000 images
* 10 different categories

These categories are:

* Airplane 
* Car 
* Bird 
* Cat 
* Deer 
* Dog 
* Frog 
* Horse 
* Ship 
* Truck 

Each image is very small (32×32 pixels), which makes it a good beginner dataset.

---

## What is a CNN?

A Convolutional Neural Network (CNN) is a type of deep learning model designed specifically for images.

It works by:

1. Detecting patterns like edges and shapes
2. Extracting important features
3. Using those features to classify the image

---

## Model Architecture

The model used in this project consists of:

* 2 Convolutional Layers (feature extraction)
* ReLU Activation Function
* Max Pooling Layers (dimensionality reduction)
* Fully Connected Layers (classification)

---

## Training Details

* Framework: PyTorch
* Optimizer: Adam
* Loss Function: CrossEntropyLoss
* Epochs: 10

---

## Results (Important)

* Final Accuracy: **72.34%**

### What does this mean?

This means:
  Out of 100 images, the model correctly predicts around **72 images**

### Loss Behavior

* Loss decreased significantly during training
* This indicates that the model was learning effectively

---

## Output Screenshot

Below is the training result showing model accuracy:

![Result Screenshot](result.png)

---

## Key Learnings

* CNNs are powerful for image classification tasks
* Pooling reduces computation and prevents overfitting
* Proper training and optimization improve performance

---

## Conclusion

In this project, I successfully built a CNN model that can classify images into 10 categories. This helped me understand how deep learning models work in real-world image classification problems.

---

## Live Blog

[Click here to view the live project](https://techie-surya.github.io/cifar10-image-classifier/)
