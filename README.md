# Multiclass Fish Image Classification
# Project Overview

This project focuses on classifying fish images into multiple species using deep learning techniques. A CNN model is built from scratch, and transfer learning is applied using multiple pre-trained models to improve accuracy and efficiency.
The best-performing model is deployed using a Streamlit web application for real-time fish image prediction.

Problem Statement

To develop a robust deep learning system capable of accurately classifying fish images into multiple categories. The project explores different CNN architectures, evaluates their performance, and deploys the best model as an interactive web application.

Skills & Technologies Used

Python

Deep Learning

TensorFlow 

Convolutional Neural Networks (CNN)

Transfer Learning

Image Preprocessing & Augmentation

Model Evaluation & Visualization

Streamlit (Model Deployment)

# Domain

 Image Classification

# Business Use Cases

Enhanced Accuracy: Identify the best-performing model for fish classification

Model Comparison: Evaluate CNN and transfer learning models using standard metrics

Deployment Ready: Provide a user-friendly web app for real-time predictions

Scalability: Reusable architecture for similar image classification tasks

# Dataset

Fish images categorized into folders by species

Loaded using TensorFlow ImageDataGenerator

Dataset provided as a ZIP file

# Project Approach
# Data Preprocessing & Augmentation

Image rescaling to [0, 1]

Data augmentation techniques:

Rotation

Zoom

Horizontal Flip

Improves generalization and reduces overfitting

# Model Training

CNN model trained from scratch

# Transfer learning using pre-trained models:

VGG16

ResNet50

MobileNet

EfficientNetB0

Fine-tuning applied to selected layers

Best-performing model saved in .h5 / .pkl format

# Model Evaluation

Models evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix


# Deployment (Streamlit App)

The Streamlit application allows users to:

Upload a fish image

Predict the fish species

View confidence scores for predictions

# Model Performance Summary
Model	Accuracy	Remarks
CNN (Scratch)	Moderate	Baseline model
VGG16	High	Heavy & slower
ResNet50	High	Stable performance
MobileNet	Fast	Lightweight & efficient
EfficientNetB0	Best	Best accuracy-speed tradeoff
# Project Deliverables

 Trained CNN & transfer learning models (.h5 / .pkl)

 Streamlit web application

 Python scripts for training, evaluation & deployment

 Model comparison report

 Well-documented GitHub repository
