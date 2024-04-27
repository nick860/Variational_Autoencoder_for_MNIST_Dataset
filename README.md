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

# Genrate 3 new images of the digit 0

![Figure_3](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/a80c37fe-58c6-4a30-95cd-57eca9a88e97)![Figure_2](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/b3d95929-95bf-4f7f-a134-85cb8d232aa3)![Figure_1](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/83c53df2-967f-4e00-902a-358b8b2270c0)

# Genrate 3 new images of the digit 2

![22](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/22962346-5528-4e7d-8747-a7dfba86edd6)![2](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/20610405-5f7c-4e92-9540-c953032411c7)![3](https://github.com/nick860/Variational_Autoencoder_for_MNIST_Dataset/assets/55057278/91ce3cfa-754e-4ce5-9b5a-80d7a6808e2e)






