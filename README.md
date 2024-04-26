# Variational Autoencoder for MNIST Dataset

This repository contains code for training and using a Variational Autoencoder (VAE) on the MNIST dataset. VAEs are generative models capable of learning latent representations of data, allowing for generation of new samples.

## Overview

The project consists of two main files:
- `train.py`: Contains code for configuring, training, and using the VAE model.
- `autoencoder.py`: Defines the architecture of the Variational Autoencoder as a PyTorch neural network model.

## Files

- `train.py`: 
  - Configures and trains the VAE model.
  - Provides functionality to generate new images based on learned latent representations.

- `autoencoder.py`:
  - Defines the architecture of the Variational Autoencoder as a PyTorch neural network model.
  - Includes the encoder and decoder components.

## Usage

1. **Training the Model**:
   - Run `train.py` to train the VAE model on the MNIST dataset.
   - Trained model parameters are saved in `model.pth`.

2. **Generating New Images**:
   - After training or loading the pre-trained model, you can generate new images by running `train.py`.
   - Enter the digit you want to generate when prompted.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib

## Citation

If you find this code useful, please consider citing:

