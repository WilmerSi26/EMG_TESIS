# =============================================================================
# SECCIÓN 1: IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization, MaxPooling1D,
                                     LSTM, Dense, Dropout)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================================================================
# SECCIÓN 2: CONFIGURACIÓN Y PARÁMETROS GLOBALES
# =============================================================================
# --- Parámetros de datos ---
DATASET_PATH = 'ninapro_db1_data'  # <-- ¡RUTA A LA CARPETA CON LOS ARCHIVOS .MAT!
SUBJECTS_TO_PROCESS = [1, 2, 3, 4, 5]  # MODIFICADO: Lista de sujetos a procesar
EXERCISES_TO_PROCESS = [1, 2, 3]  # Ejercicios a incluir para cada sujeto

# --- Parámetros de preprocesamiento ---
WINDOW_SIZE = 200
STEP = 50

# --- Parámetros del modelo y entrenamiento ---
NUM_CLASSES = 53
EPOCHS = 100  # Aumentado para dar margen al EarlyStopping
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2


# =============================================================================
# SECCIÓN 3: FUNCIONES DE CARGA Y PREPROCESAMIENTO (Sin cambios)
# =============================================================================
def load_ninapro_data(base_path, subject, exercises):
    all_emg = np.array([])
    all_gestures = np.array([])

    for exercise in exercises:
        file_path = os.path.join(base_path, f'S{subject}_A1_E{exercise}.mat')
        try:
            data = loadmat(file_path)
            if all_emg.size == 0:
                all_emg = data['emg']
                all_gestures = data['restimulus']
            else:
                all_emg = np.vstack((all_emg, data['emg']))
                all_gestures = np.vstack((all_gestures, data['restimulus']))
        except FileNotFoundError:
            print(f"¡ADVERTENCIA! No se encontró el archivo: {file_path}")
            return None, None

    return all_emg, all_gestures


def create_windows(emg, gestures, window_size, step):
    X, y = [], []
    active_indices = np.where(gestures.flatten() != 0)[0]

    for i in range(0, len(active_indices) - window_size, step):
        window_indices = active_indices[i: i + window_size]
        if window_indices[-1] - window_indices[0] != window_size - 1:
            continue

        window_emg = emg[window_indices]
        window_gestures = gestures[window_indices]
        label = np.bincount(window_gestures.flatten()).argmax()

        mean = np.mean(window_emg, axis=0)
        std = np.std(window_emg, axis=0)
        window_normalized = (window_emg - mean) / (std + 1e-8)

        X.append(window_normalized)
        y.append(label)

    return np.array(X), np.array(y)


# =============================================================================
# SECCIÓN 4: FUNCIÓN PARA CONSTRUIR EL MODELO (Sin cambios)
# =============================================================================
def build_hybrid_cnn_lstm_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)
    cnn = Conv1D(filters=64, kernel_size=9, activation='relu', padding='same')(input_layer)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.3)(cnn)
    cnn = Conv1D(filters=128, kernel_size=9, activation='relu', padding='same')(cnn)
    cnn = BatchNormalization()(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Dropout(0.3)(cnn)
    lstm = LSTM(128, return_sequences=False)(cnn)
    lstm = Dropout(0.4)(lstm)
    output_layer = Dense(num_classes, activation='softmax')(lstm)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# =============================================================================
# SECCIÓN 5: SCRIPT PRINCIPAL DE EJECUCIÓN (MODIFICADO)
# =============================================================================
if __name__ == "__main__":
    # --- MODIFICADO: Bucle para procesar múltiples sujetos ---
    all_X = []
    all_y = []

    for subject_id in SUBJECTS_TO_PROCESS:
        print(f"--- Cargando y procesando Sujeto {subject_id} ---")
        emg_signals, gesture_labels = load_ninapro_data(DATASET_PATH, subject_id, EXERCISES_TO_PROCESS)
        if emg_signals is None:
            print(f"Omitiendo sujeto {subject_id} por falta de datos.")
            continue

        print(f"Creando ventanas para el sujeto {subject_id}...")
        X, y = create_windows(emg_signals, gesture_labels, WINDOW_SIZE, STEP)
        all_X.append(X)
        all_y.append(y)

    # Combinar datos de todos los sujetos en un solo gran array
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)

    print("\n--- Datos de todos los sujetos combinados ---")
    print(f"Forma total de X: {X_combined.shape}")
    print(f"Forma total de y: {y_combined.shape}")

    # Preprocesamiento de etiquetas
    y_combined = y_combined - 1
    y_categorical = to_categorical(y_combined, num_classes=NUM_CLASSES - 1)

    # División en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_categorical, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_combined
    )

    print(f"\nDatos listos para el entrenamiento. Formas:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - y_train: {y_train.shape}")
    print(f"  - X_val: {X_val.shape}")
    print(f"  - y_val: {y_val.shape}")

    # Construir el modelo
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_hybrid_cnn_lstm_model(input_shape, num_classes=NUM_CLASSES - 1)
    model.summary()

    # --- MODIFICADO: Definir callbacks para el entrenamiento ---
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    MODEL_CHECKPOINT_FILENAME = 'mejor_modelo_multisujeto.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_CHECKPOINT_FILENAME,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Entrenar el modelo
    print("\nIniciando entrenamiento (con Early Stopping)...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_callback, model_checkpoint_callback]
    )

    # Visualizar resultados del entrenamiento
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo (Multi-Sujeto)')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo (Multi-Sujeto)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Guardar el modelo final (que ya tiene los mejores pesos gracias a restore_best_weights)
    FINAL_MODEL_FILENAME = 'modelo_final_multisujeto.h5'
    model.save(FINAL_MODEL_FILENAME)
    print(f"\nModelo final (mejor versión) guardado como '{FINAL_MODEL_FILENAME}'")
    print(f"El mejor modelo también se guardó como '{MODEL_CHECKPOINT_FILENAME}' durante el entrenamiento.")



