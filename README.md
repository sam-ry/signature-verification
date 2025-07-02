# Signature Verification with Siamese Network

This project uses deep learning to check if two signature images are from the same person or if one is a forgery. **Siamese Neural Network** is trained on a dataset of real and fake signatures.

## What It Does

* Takes **two signature images** as input.
* Compares them using a trained model.
* Predicts if the signatures are **genuine** or **forged**.

## Dataset
* Positive pairs: genuine signatures from the same user.
* Negative pairs: genuine and one forged signatures.

## Model Details

### 1. **Backbone (Feature Extractor)**

* A small CNN (Convolutional Neural Network).
* Converts each image into a feature vector.
* Normalizes the output using L2 normalization.

### 2. **Siamese Network**

* Takes two images.
* Passes both through the same CNN.
* Compares the feature vectors using L1 distance.
* Final layer predicts similarity score (0 to 1).

## How It Works

1. Train the model using the signature dataset.
2. Save the model after training.
3. Use a function to compare any two images.
4. Output: Prediction: Genuine or Forged
