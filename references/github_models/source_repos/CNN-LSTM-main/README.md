# EMG Gesture Recognition using CNN-LSTM Hybrid Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
  [![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?logo=Keras&logoColor=white)](https://keras.io)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Downloads](https://static.pepy.tech/badge/your-package-name)](https://pepy.tech/project/your-package-name)

A hybrid CNN-LSTM model for sEMG-based gesture recognition using the NinaPro DB1 dataset.

## 📌 Overview

This repository contains a deep learning pipeline for surface electromyography (sEMG) signal classification using a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture. The model is trained on the [NinaPro DB1 dataset](http://ninapro.hevs.ch/) to recognize hand gestures from EMG signals.

![Model Architecture](https://via.placeholder.com/800x400.png?text=CNN-LSTM+Architecture+Diagram)

## ✨ Features

- **Hybrid Architecture**: Combines CNN's feature extraction with LSTM's temporal modeling
- **Data Preprocessing**: Sliding window approach with Z-score normalization
- **Modular Design**: Clean separation of data loading, preprocessing, and model building
- **Visualization**: Training history and performance metrics plotting

## 📊 Dataset

The model uses the [NinaPro DB1 dataset](http://ninapro.hevs.ch/) which contains:
- 27 healthy subjects
- 52 hand gestures + rest position
- 10 EMG electrodes at 100Hz sampling rate

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CNN-LSTM.git
cd emg-gesture-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
Download the NinaPro DB1 dataset and place it in the ninapro_db1_data folder
Run the main script:

``` python
python emg_gesture_recognition.py
Configuration Options
Modify these parameters in the script:
```
```python
# --- Data parameters ---
SUBJECT_ID = 1                     # Subject to process (1-27)
EXERCISES_TO_PROCESS = [1, 2, 3]   # Exercises to include

# --- Preprocessing ---
WINDOW_SIZE = 200                  # Window length in samples (~200ms at 100Hz)
STEP = 50                          # Window step size

# --- Training ---
EPOCHS = 30
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
📈 Results
Example performance on Subject 1 (Exercises 1-3):

Metric	Training	Validation
Accuracy	92.4%	85.7%
Loss	0.21	0.48
https://via.placeholder.com/600x300.png?text=Training+and+Validation+Curves
```
## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References
Atzori, M., et al. (2014). "The NinaPro database: Evaluation of hand movement recognition methods with machine learning techniques."

Geng, W., et al. (2016). "A novel hybrid CNN-LSTM scheme for sEMG-based gesture recognition."


Remember to:

Replace placeholder images with actual screenshots/plots from your project

Update the DOI badge with your actual Zenodo DOI if you have one

Customize the configuration options to match your actual script parameters

Add your actual performance metrics when available
