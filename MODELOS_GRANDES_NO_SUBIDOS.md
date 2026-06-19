# Modelos grandes no versionados

Los siguientes archivos superan el limite recomendado para GitHub y no se suben al repositorio:

- `models/baseline/emg_sklearn_baseline.joblib`
- `models/baseline_final/emg_sklearn_baseline.joblib`
- `models/baseline_realtime/emg_sklearn_baseline.joblib`
- `models/rf_final/emg_rf_final.joblib`

Estos archivos corresponden a modelos Random Forest serializados. Se excluyen porque cada uno pesa varios cientos de MB y GitHub bloquea archivos mayores a 100 MB cuando no se usa Git LFS.

Para reproducirlos:

```powershell
.\.venv\Scripts\python.exe src\training\train_baseline_rf.py
```

Para trabajar con TinyML y XIAO ESP32S3, el artefacto principal versionado es:

```text
models/tiny_mlp_final/emg_tiny_mlp_esp32_compatible.tflite
```

El firmware Arduino usa el encabezado generado en:

```text
firmware/xiao_esp32s3_tinyml_inference/generated/model_data.h
```
