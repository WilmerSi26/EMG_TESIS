# Flujo defendible de entrenamiento, validacion e inferencia

## 1. Auditoria del dataset

Primero se revisan los CSV adquiridos para eliminar tomas incompletas. En este proyecto se revisaron 37 archivos y se conservaron 34 archivos validos de 4 sujetos: `S001`, `S002`, `S003` y `S004`.

Comando:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\preprocessing\audit_final_dataset.py --copy-valid --clear-valid-dir
```

Que hace:

- Verifica que existan los 4 canales `emg_ch1` a `emg_ch4`.
- Verifica frecuencia de muestreo de 200 Hz.
- Revisa que el archivo tenga las 10 clases funcionales.
- Excluye archivos cortados o solo de preparacion/reposo.
- Copia los CSV validos a `data\final_10clases_valid`.

Archivos generados:

- `results\metrics\final_dataset_audit.csv`
- `results\reports\final_dataset_audit.md`

## 2. Segmentacion en ventanas

Las senales continuas se dividen en ventanas de 40 muestras con salto de 10 muestras.

- 40 muestras a 200 Hz equivalen a 0.2 s.
- 10 muestras a 200 Hz equivalen a una actualizacion cada 0.05 s.

Esto permite reconocer gestos sin esperar varios segundos.

## 3. Extraccion de caracteristicas

Por cada ventana se calculan 7 caracteristicas por canal:

- media
- desviacion estandar
- minimo
- maximo
- media absoluta
- RMS
- longitud de forma de onda

Como son 4 canales, se obtienen 28 caracteristicas por ventana. Esta es la entrada usada por RandomForest y Tiny MLP.

## 4. Validacion aleatoria

La validacion aleatoria mezcla ventanas del conjunto total y separa 80 % para entrenamiento y 20 % para prueba.

Resultado actual:

- RandomForest: 78.77 %
- Tiny MLP TFLite: 62.70 %

Interpretacion:

Esta prueba demuestra que las senales adquiridas tienen separabilidad dentro del conjunto de datos. Es util para validar el brazalete, la adquisicion y el flujo de entrenamiento.

## 5. Validacion dejando un sujeto fuera

La validacion Leave-One-Subject-Out entrena con tres sujetos y prueba con el sujeto restante.

Resultado actual:

| Sujeto evaluado | Exactitud |
|---|---:|
| S001 | 17.37 % |
| S002 | 16.26 % |
| S003 | 51.20 % |
| S004 | 39.62 % |

Exactitud media: 31.11 %.

Interpretacion:

Esta prueba es mas exigente porque las senales sEMG cambian entre personas por anatomia, posicion del brazalete, impedancia de contacto, fuerza de contraccion y ruido. Por eso, para control protesico real se justifica calibrar el modelo con el usuario final o reportar una etapa de adaptacion por usuario.

## 6. Inferencia en computador

La inferencia en computador usa el modelo RandomForest entrenado con los CSV validos. Lee la XIAO por serial, grafica la senal en vivo y muestra la clase inferida.

Comando:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\inference\live_inference_rf.py --port COM4 --model-dir models\baseline_final
```

Que hace:

- Recibe muestras seriales desde la XIAO.
- Forma ventanas de 40 muestras.
- Calcula las 28 caracteristicas.
- Escala las caracteristicas con el mismo `StandardScaler` del entrenamiento.
- Predice la clase con RandomForest.
- Muestra la clase y confianza en tiempo real.

## 7. Inferencia en XIAO

Para la XIAO se usa el Tiny MLP TFLite porque es pequeno y embebible.

Primero se generan cabeceras Arduino:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\embedded\export_tiny_mlp_for_xiao.py --model-dir models\tiny_mlp_final
```

Luego se compila:

```text
firmware\xiao_esp32s3_tinyml_inference\xiao_esp32s3_tinyml_inference.ino
```

Que hace el firmware:

- Lee los 4 sensores a 200 Hz.
- Calcula las mismas 28 caracteristicas que Python.
- Normaliza con la media y escala del entrenamiento.
- Ejecuta el modelo TFLite.
- Imprime `INFER,<indice>,<etiqueta>,<confianza>`.

Para leer la salida de la XIAO:

```powershell
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe src\inference\read_xiao_inference.py --port COM4
```

## 8. Como defender los dos resultados

No se debe decir solamente "el modelo tiene 78.77 %". La defensa correcta es:

"La validacion aleatoria alcanzo 78.77 % con RandomForest, lo que confirma separabilidad dentro del conjunto adquirido. Sin embargo, al dejar un sujeto completamente fuera, la exactitud media baja a 31.11 %, mostrando alta variabilidad inter-sujeto. Por esta razon, el sistema se plantea con una fase de calibracion/entrenamiento por usuario antes del despliegue embebido."

Esa lectura es tecnicamente mas fuerte y evita prometer generalizacion universal con solo cuatro participantes.
