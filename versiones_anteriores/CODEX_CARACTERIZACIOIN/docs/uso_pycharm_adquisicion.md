# Uso en PyCharm Community para adquirir senales

## Forma mas simple

Ejecuta:

```text
src/acquisition/acquire_movements.py
```

Si solo hay un puerto COM, el script lo usara automaticamente. Si hay varios,
mostrara una lista y pedira escribir el numero del puerto.

## Forma recomendada con parametros

En PyCharm:

1. Abre `Run > Edit Configurations`.
2. Selecciona la configuracion de `acquire_movements.py`.
3. En `Script parameters`, coloca algo como:

```text
--port COM5 --subject-id S001 --session-id SES001 --repetitions 5
```

Cambia `COM5` por el puerto real de tu XIAO ESP32S3.

## Ver puertos

Puedes ejecutar con:

```text
--list-ports
```

o desde terminal:

```powershell
python src\acquisition\acquire_movements.py --list-ports
```

## Rutina por defecto

Al ejecutar sin parametros avanzados, se registra:

- Sujeto: `S001`.
- Repeticiones: `5`.
- Movimiento: `4 s`.
- Reposo: `2 s`.
- Movimientos: `cierre`, `apertura`, `pinza`.

Los CSV se guardan en:

```text
data/raw/
```
