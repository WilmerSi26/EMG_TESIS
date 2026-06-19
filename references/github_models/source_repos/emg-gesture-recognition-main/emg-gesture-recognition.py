"""
EMG Gesture Recognition using CNN

This script implements a CNN-based model for classifying hand gestures
from multi-channel surface EMG recordings (NinaPro-style data).

Note:
- This is an exploratory research implementation.
- Performance is sensitive to preprocessing and label definitions.

Authors: Isha Yeleswarapu, Justine Choueiri, Tvisha Nepani
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch

# -------------------------
# Signal Processing
# -------------------------

def notch_filter(data, f0=50.0, fs=2000.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data, axis=0)

def bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=2000.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype="band")
    return filtfilt(b, a, data, axis=0)

def normalize(data):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

# -------------------------
# Data Loading
# -------------------------

def load_ninapro(paths, max_len=4000):
    """
    Loads EMG and restimulus labels from NinaPro .mat files.
    Returns fixed-length windows and integer-mapped labels.
    """
    X, y = [], []

    for path in paths:
        data = loadmat(path)
        emg = normalize(bandpass_filter(notch_filter(data["emg"])))
        labels = data["restimulus"].flatten()

        for i in range(0, len(emg) - max_len, max_len):
            window = emg[i:i+max_len, :12]
            label = labels[i]

            if label > 0:
                X.append(window)
                y.append(label)

    # Map labels to contiguous integers
    unique_labels = sorted(set(y))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])

    return np.array(X), y, label_map

# -------------------------
# Model
# -------------------------

class CNN(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

# -------------------------
# Training Loop
# -------------------------

def train_model(model, train_loader, test_loader, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for X, y in test_loader:
                preds.extend(torch.argmax(model(X), dim=1).tolist())
                truths.extend(y.tolist())

        print("Predicted class distribution:", np.bincount(preds))
        acc = accuracy_score(truths, preds)
        print(f"Epoch {epoch+1}: accuracy = {acc:.3f}")

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    paths = [
        "data/S1_E1_A1.mat"
    ]

    try:
        X, y, label_map = load_ninapro(paths)
    except FileNotFoundError:
        print("NinaPro data not found. Please place .mat files in ./data/")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    model = CNN(channels=12, num_classes=len(label_map))
    train_model(
        model,
        DataLoader(train_ds, batch_size=16, shuffle=True),
        DataLoader(test_ds, batch_size=16),
        epochs=5
    )
