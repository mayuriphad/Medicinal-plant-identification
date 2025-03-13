# 🌿 Medicinal Plant Identification System

## Overview

An advanced computer vision application that identifies medicinal plants from images using deep learning. Built with a fine-tuned MobileNetV2 architecture and deployed as a user-friendly web application.


## 🎯 Key Features

- **High Accuracy Classification**: Identifies 40 different medicinal plant species with 93.7% validation accuracy
- **User-Friendly Interface**: Simple upload-and-identify web application
- **Educational Resources**: Automatically provides relevant information about identified plants
- **Mobile-Optimized Model**: Uses MobileNetV2 architecture for efficient inference

## 📊 Dataset

The system is trained on the [Indian Medicinal Plant Image Dataset](https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset), which includes:

- **40 Classes** of medicinal plants native to India
- **Images standardized to 224×224 pixels**
- **Data split**: 80% training / 20% validation
- **~200-300 images per class** for robust training

## 🧠 Model Architecture

### Base Model
- **MobileNetV2** (pre-trained on ImageNet)
- Optimized for mobile and edge devices
- Excellent balance between accuracy and computational efficiency

### Custom Classification Head
```
GlobalAveragePooling2D
↓
Dense (256 units, ReLU activation)
↓
Dropout (0.5)
↓
Dense (40 units, Softmax activation)
```

### Training Strategy
1. **Initial Training Phase**: Frozen base model with trainable classification head (20 epochs)
2. **Fine-Tuning Phase**: Last 20 layers unfrozen for further optimization (10 epochs)

### Hyperparameters
| Parameter | Initial Training | Fine-Tuning |
|-----------|------------------|-------------|
| Learning Rate | 0.001 | 0.00001 |
| Batch Size | 32 | 32 |
| Optimizer | Adam | Adam |
| Loss Function | Categorical Cross-Entropy | Categorical Cross-Entropy |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Training Accuracy | 95.3% |
| Validation Accuracy | 93.7% |
| Training Loss | 0.18 |
| Validation Loss | 0.25 |

## 📱 Web Application

A Flask-based web interface allows users to easily interact with the model:

1. **Upload an image** of an unknown medicinal plant
2. **Get instant identification** of the plant species
3. **Access educational resources** via automated search results

### Tech Stack
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Model Serving**: TensorFlow/Keras
- **Additional API**: Google Custom Search for information retrieval

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10.9
- TensorFlow 
- Flask
- Other dependencies in `requirements.txt`

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/Medical-plant-identification.git
cd Medical-plant-identification

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then visit `http://localhost:5000` in your browser.


## 📊 Data Augmentation Techniques

To improve model robustness, the following augmentations were applied during training:

- Rescaling (1/255)
- Random rotation (±20°)
- Shear transformation (0.2)
- Zoom range (0.2)
- Horizontal flip
- Width and height shift (0.2)

## 🔄 Prediction Workflow

1. **Image Preprocessing**:
   - Resize to 224×224 pixels
   - Normalize pixel values

2. **Model Inference**:
   - Forward pass through MobileNetV2
   - Extract class probabilities

3. **Result Processing**:
   - Select highest probability class
   - Retrieve plant information
   - Generate search results

## 📋 Supported Plant Species

The system can identify 40 medicinal plants including:
- Tulasi
- Neem
- Aloe Vera
- Turmeric
- Ashwagandha
- And many more...

## 🛠️ Future Improvements

- [ ] Expand the dataset to include more plant species
- [ ] Implement explainable AI techniques for visualization
- [ ] Create a mobile application version
- [ ] Add more detailed plant information database
- [ ] Implement offline mode for field use

## 📝 Citation

If you use this project in your research, please cite:

```
@software{medicinal_plant_identification,
  author = {Mayuri Phad},
  title = {Medicinal Plant Identification System},
  year = {2025},
  url = {https://github.com/mayuriphad/Medicinal-plant-identification-.git}
}
```



## 🤝 Acknowledgements

- [Indian Medicinal Plant Image Dataset](https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset) for the training data
- TensorFlow and Keras teams for the deep learning framework
- Flask team for the web framework