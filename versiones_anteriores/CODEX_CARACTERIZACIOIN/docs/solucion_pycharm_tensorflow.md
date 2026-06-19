# Solucion en PyCharm Community: error instalando TensorFlow

## Error observado

PyCharm intento instalar:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv\Scripts\python.exe
```

y pip descargo paquetes `cp314`, por ejemplo:

```text
numpy-...-cp314-cp314-win_amd64.whl
```

Eso significa que el entorno virtual fue creado con Python 3.14.

El error:

```text
ERROR: No matching distribution found for tensorflow
```

ocurre porque no existe una rueda compatible de TensorFlow para esa version de
Python en tu entorno.

## Solucion rapida para seguir adquiriendo senales

Instala solo las dependencias base:

```powershell
cd D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` ya no incluye TensorFlow. Con eso puedes:

- Leer puerto serial.
- Guardar CSV.
- Ver graficas.
- Preparar datos.

## Solucion para entrenar la RNN

Para entrenar, crea un entorno nuevo con Python 3.11 o Python 3.12 de 64 bits.

En PyCharm:

1. `File > Settings > Project > Python Interpreter`.
2. Abre el menu del interprete.
3. Selecciona `Add Interpreter`.
4. Crea un `Virtualenv`.
5. En `Base interpreter`, selecciona Python 3.11 o 3.12.
6. Cuando se cree el nuevo entorno, instala:

```powershell
python -m pip install --upgrade pip
pip install -r requirements-training.txt
```

## Comprobar version de Python

Dentro de la terminal de PyCharm:

```powershell
python --version
```

Debe mostrar algo como:

```text
Python 3.11.x
```

o:

```text
Python 3.12.x
```

Evita Python 3.14 para el entrenamiento con TensorFlow.
