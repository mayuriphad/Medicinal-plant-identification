# Medical-plant-identification-
# Description

# Medicinal Plant Identification System
This repository contains the complete implementation of a Medicinal Plant Identification System using a pre-trained MobileNetV2 model. The system identifies medicinal plants from images and provides basic information about them using a simple web application built with Flask.

# Dataset
Indian Medicinal Plant Image Dataset: https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset

Number of Classes: 40
Image Size: 224x224
Dataset Split: 80% training, 20% validation

# Model Architecture

The system employs a pre-trained MobileNetV2 model, fine-tuned for the task of multi-class classification of medicinal plant images.

# Layers
Base Model: MobileNetV2 (pre-trained on ImageNet, without top layers)
Custom Layers:
GlobalAveragePooling2D
Dense (256 units, ReLU activation)
Dense (40 classes, softmax activation)

# Model Compilation
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Performance Metrics
Training Accuracy: 95.3%
Validation Accuracy: 93.7%
Training Loss: 0.18
Validation Loss: 0.25

# System Overview

Web Application
The Flask-based web application allows users to upload an image of a medicinal plant and obtain predictions.

Input: An image of a medicinal plant.

Output: Predicted plant name and relevant links for additional information.

# Web Interface

Upload Image: Users can upload an image of the medicinal plant.
Prediction Result: Displays the predicted plant name.
Search Results: Links to additional information about the plant.

# Sample Prediction Flow

Upload: User uploads an image of a plant (e.g., Tulasi).
Prediction: The system predicts the class label (Tulasi).
Search Results: The system fetches top 3 Google search results for "Information about Tulasi."

# Model Training

The training process involves fine-tuning the MobileNetV2 model on the dataset with data augmentation techniques.

Data Augmentation
Rescaling (1/255)
Shear Transformation
Zoom
Horizontal Flip
Training Phases

Base Model Frozen: Trained only custom layers (20 epochs).

Training Accuracy: 91.4%
Validation Accuracy: 89.7%

Fine-Tuning: Unfroze the last 20 layers for fine-tuning (10 epochs).
Training Accuracy: 95.3%
Validation Accuracy: 93.7%

Hyperparameters

Learning Rate: 0.001 (initial), 0.00001 (fine-tuning)

Batch Size: 32

Model Save Path

C:\Users\Mayuri\Desktop\Medical plant identification\medicinal_plant_classifier.h5

# Deployment

Run the Flask Application
python app.py
Access the Application
Open http://localhost:5000 in a web browser.

# Conclusion

This system successfully identifies medicinal plants from images with high accuracy, leveraging the MobileNetV2 model. The integration of a user-friendly Flask interface and Google Search API enhances the system's utility for educational and research purposes.