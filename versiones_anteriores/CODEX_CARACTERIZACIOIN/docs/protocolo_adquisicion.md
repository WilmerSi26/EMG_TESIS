# Protocolo de adquisicion

## Objetivo

Registrar senales sEMG multicanal para caracterizar tres movimientos:

- `cierre`
- `apertura`
- `pinza`

## Preparacion

1. Verificar que los 4 sensores esten alimentados a 3.3 V y compartan GND con
   la XIAO ESP32S3.
2. Ubicar los electrodos siempre en la misma zona muscular para una sesion.
3. Esperar 1 a 2 minutos despues de colocar electrodos para estabilizar la
   lectura.
4. Evitar mover cables durante la toma.
5. No activar actuadores ni servos durante la adquisicion de entrenamiento.

## Rutina sugerida

Por cada repeticion:

1. Reposo.
2. Cierre de mano.
3. Reposo.
4. Apertura de mano.
5. Reposo.
6. Pinza fina.

Parametros iniciales:

- Reposo: 2 s.
- Movimiento: 4 s.
- Repeticiones por sesion: 5 a 10.
- Sesiones por sujeto: 3 o mas.
- Frecuencia de muestreo: 1000 Hz si el enlace serial es estable, 500 Hz si hay
  perdida de datos.

## Etiquetas validas

Los scripts de entrenamiento normalizan estas etiquetas:

```text
cierre, cierre_mano
apertura, apertura_mano
pinza, pinza_fina
```

Las fases `reposo`, `preparacion` o `pausa` se guardan para trazabilidad, pero
no se usan por defecto para entrenar el clasificador de tres movimientos.

## Calidad de datos

Descartar o repetir una sesion si ocurre alguno de estos casos:

- Canal saturado cerca de 0 o 4095 durante mucho tiempo.
- Desconexion de electrodo.
- Movimiento equivocado durante una etiqueta.
- Perdida visible de muestras o congelamiento de la grafica.
- Mucho ruido por cables sueltos o mala referencia de tierra.
