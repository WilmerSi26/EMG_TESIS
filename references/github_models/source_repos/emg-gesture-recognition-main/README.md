# EMG Gesture Recognition with CNN-LSTM
CNN-LSTM model for EMG-based hand gesture classification

Overview:
This project explores the preprocessing, organization, and modeling of surface EMG signals for gesture recognition. Using the NinaPro dataset, we build a pipeline to:
- Filter and normalize raw EMG signals
- Split signals by exercise/rest periods
- Train a simple CNN-LSTM model to classify hand gestures
The goal of this project was educational and exploratory: to demonstrate EMG data handling, PyTorch modeling, and pipeline structuring rather than achieving state-of-the-art performance.

Features:
EMG preprocessing
- Notch filtering to remove line noise
- Bandpass filtering to isolate EMG signal frequencies
- Normalization per channel
Data organization
- Splitting signals by exercise/rest
- Preparing train/test sets for sequence classification
- CNN-LSTM model for sequence classification
- Training loop with validation accuracy tracking
- Simple analysis of predicted class distribution to identify model behavior

Installation:
Clone the repository
```python
git clone https://github.com/YOUR_USERNAME/emg-gesture-recognition.git
cd emg-gesture-recognition
```

Usage:
Place your .mat NinaPro EMG data files in a local folder.
Update the paths variable in emg_gesture_recognition.py with the file paths to your .mat files.
Run the script.
```python
python3 emg_gesture_recognition.py
```

The script will train a CNN-LSTM model and print
- Training loss per epoch
- Training and validation accuracy
- Predicted class distributions for insight into model behavior

Model Behavior & Limitations:
The CNN-LSTM model trained on this dataset exhibits class collapse, predicting mostly a single gesture class.
Accuracy remains low (~1–5%), which is consistent with challenges in EMG gesture classification
- Class imbalance and limited data per gesture
- Inter-subject variability
 Minimal hyperparameter tuning in this exploratory project
The purpose of this repository is to showcase preprocessing, data handling, and pipeline construction rather than to produce a high-performing model.

Future Improvements: 
Explore LSTM hyperparameters, data augmentation, class balancing techniques, and alternative architectures.
