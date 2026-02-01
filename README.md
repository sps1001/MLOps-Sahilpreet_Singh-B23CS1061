# Lab 2 Worksheet - CIFAR-10 CNN Classification

## Overview
This project implements a Convolutional Neural Network (CNN) for CIFAR-10 image classification, featuring a custom dataloader, FLOPs calculation, gradient flow visualization, and comprehensive training monitoring using Weights & Biases (Wandb).

## Model Architecture
- **CNN Architecture**: 
  - Conv2d layers: 3→32→64 channels with 3×3 kernels
  - MaxPool2d layers for downsampling
  - Fully connected layers: 4096→256→10
  - ReLU activation functions
- **Total Parameters**: 1,070,794
- **FLOPs (MACs)**: 6,654,464

## Custom Dataloader
Implemented `CIFAR10Custom` class that:
- Wraps the CIFAR-10 dataset with custom transformations
- Applies data augmentation for training (RandomHorizontalFlip, RandomCrop)
- Normalizes images to [-1, 1] range
- Supports both training and test splits

## Training Configuration
- **Dataset**: CIFAR-10 (50,000 train, 10,000 test images)
- **Epochs**: 25
- **Batch Size**: 128
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: CrossEntropyLoss
- **Device**: CUDA (GPU accelerated)

## Visualizations & Monitoring
- **Gradient Flow**: Real-time visualization of average gradients across all layers during training
- **Wandb Integration**: 
  - Training loss and accuracy tracking
  - Test accuracy logging
  - Project: `cnn-cifar10-lab2`

## Results
- **Training Accuracy**: 79.16% (after 25 epochs)
- **Test Accuracy**: 78.07%
- **Final Training Loss**: 233.56

## Key Observations
1. Model achieved steady convergence with consistent accuracy improvement over 25 epochs
2. Gradient flow visualization confirmed healthy gradient propagation throughout training
3. Training and test accuracies are closely aligned, indicating good generalization
4. The model architecture provides a good balance between complexity and performance

## Setup Instructions
```bash
pip install wandb thop torchvision matplotlib torch
```

## Files
- `Lab2_Assignment1.ipynb`: Complete implementation notebook with all code, training, and visualizations

## Submission Details
- **Project**: Lab 2 Worksheet
- **Date**: 31 Jan 2026
- **Dataset**: CIFAR-10
- **Visualizations**: Available on Wandb dashboard
