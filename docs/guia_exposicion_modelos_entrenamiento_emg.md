# Guia para exposicion: modelos de entrenamiento EMG y redes neuronales

Proyecto: brazalete sEMG + XIAO ESP32S3 + clasificacion de gestos para protesis

Fecha de preparacion: 19/06/2026

## 1. Resumen ejecutivo para exposicion

Esta guia esta preparada para explicar, defender y entender la parte de entrenamiento e inteligencia artificial del sistema de tesis. El proyecto no se limita a entrenar una red neuronal aislada: primero adquiere senales sEMG desde un brazalete de 4 canales, organiza esas senales en ventanas temporales, etiqueta cada ventana con un gesto funcional y compara varios modelos para decidir cuales sirven en computador y cuales son candidatos para migrar a la XIAO ESP32S3.

La idea central es sencilla: cada gesto de la mano deja una huella electrica diferente en los musculos del antebrazo. Esa huella no es una linea perfecta ni una orden digital directa; es una senal analogica variable, con ruido, con diferencias entre personas y con cambios segun la posicion del brazalete. Por eso se necesita un modelo de aprendizaje automatico: el modelo aprende patrones repetidos en los canales EMG y devuelve una clase como reposo, cierre, apertura, pinza o movimiento por grupo de dedos.

La forma correcta de explicarlo es: se utilizaron arquitecturas de referencia de la literatura y de repositorios publicos, pero fueron adaptadas al protocolo propio del proyecto. No se copiaron pesos entrenados; se reentrenaron los modelos con el dataset local de 10 clases, 4 canales, 34 CSV validos, 4 sujetos, muestreo de 200 Hz, ventanas de 40 muestras y desplazamiento de 10 muestras.

## 2. Contexto del sistema de tesis

El sistema completo sigue esta cadena: musculo del antebrazo -> sensor sEMG -> XIAO ESP32S3 -> computador para adquisicion y entrenamiento -> modelo entrenado -> inferencia -> activacion de servomotores.

El brazalete utiliza cuatro sensores sEMG ubicados sobre zonas musculares del antebrazo asociadas a extension y flexion de dedos. Cada canal entrega una senal analogica procesada por el modulo EMG, y la XIAO ESP32S3 digitaliza esos valores mediante sus entradas ADC. El computador recibe las muestras por comunicacion serial, las guarda como CSV y posteriormente usa esos datos para entrenar modelos.

El protocolo actual trabaja con 10 clases funcionales: reposo, apertura, cierre, pulgar_apertura, pulgar_cierre, indice_medio_apertura, indice_medio_cierre, anular_menique_apertura, anular_menique_cierre y pinza. La reduccion a clases funcionales es importante porque una protesis real no necesita distinguir todos los dedos de manera aislada en todos los casos; necesita movimientos utiles, repetibles y viables para un usuario, especialmente si luego se prueba con un paciente o usuario del grupo de investigacion.

A 200 Hz, una ventana de 40 muestras representa aproximadamente 200 ms de senal. El stride de 10 muestras equivale a una actualizacion cada 50 ms. En palabras simples: el modelo mira pequenos fragmentos de senal de 0.2 segundos y va actualizando la prediccion varias veces por segundo.


## 3. Flujo de entrenamiento aplicado

El flujo real implementado en el proyecto puede explicarse en nueve pasos:

1. Adquisicion: la XIAO lee los 4 canales EMG y envia las muestras al computador.
2. Etiquetado: cada segmento del CSV queda asociado a un movimiento, por ejemplo cierre o pinza.
3. Limpieza de etiquetas: el sistema normaliza nombres equivalentes. Por ejemplo, abrir y apertura_mano se unifican como apertura.
4. Segmentacion: se separan los tramos consecutivos que tienen la misma etiqueta. Esto evita crear ventanas mezcladas entre, por ejemplo, reposo y cierre.
5. Ventaneo deslizante: se crean fragmentos de 40 muestras. Cada fragmento conserva los 4 canales.
6. Preprocesamiento: los modelos profundos usan normalizacion por media y desviacion estandar; RandomForest y Tiny MLP usan caracteristicas estadisticas por ventana.
7. Division de datos: se separa una parte para entrenamiento y otra para prueba. Tambien se usa validacion durante el entrenamiento.
8. Entrenamiento: el modelo ajusta sus parametros para reducir el error entre la clase real y la clase predicha.
9. Evaluacion: se generan metricas como exactitud, reporte de clasificacion y matriz de confusion.

Ejemplo cotidiano: imagina que quieres reconocer personas por su forma de caminar. No miras una foto aislada del pie; miras un fragmento corto del movimiento. Aqui pasa algo parecido: el modelo no mira una muestra ADC aislada, mira una ventana de varias muestras para reconocer el patron muscular.

## 4. Conceptos base de redes neuronales y aprendizaje automatico

Una red neuronal artificial es un conjunto de operaciones matematicas organizadas por capas. Cada capa transforma los datos de entrada para extraer informacion cada vez mas util. En este proyecto, la entrada puede ser una ventana de senales EMG y la salida es una probabilidad para cada gesto.

Neurona artificial: es una unidad matematica que recibe varios numeros, los multiplica por pesos, suma un sesgo y pasa el resultado por una funcion de activacion. No es una neurona biologica real; es una analogia. Ejemplo: si una neurona recibe los valores de energia de varios canales EMG, puede activarse mas cuando detecta una combinacion parecida a cierre.

Peso: es la importancia que el modelo asigna a una entrada. Si un canal aporta mucho para reconocer pinza, el entrenamiento puede aumentar el peso asociado a ese canal o patron.

Sesgo o bias: es un ajuste adicional que permite que una neurona no dependa solo de la multiplicacion de entradas. Es como mover el umbral de decision.

Funcion de activacion: introduce no linealidad. Sin activaciones, muchas capas juntas se comportarian como una sola operacion lineal. En los modelos se usa ReLU, que deja pasar valores positivos y corta valores negativos. Ejemplo: funciona como un filtro que dice: si esta caracteristica no aparece, la dejo en cero; si aparece, la transmito.

Capa densa: cada neurona se conecta con todas las salidas de la capa anterior. Es util para combinar caracteristicas ya extraidas y tomar una decision.

Softmax: convierte la salida final en probabilidades. Si el modelo devuelve cierre 0.78, apertura 0.10 y reposo 0.04, significa que considera mas probable la clase cierre.

Loss o funcion de perdida: mide que tan mal esta prediciendo el modelo. Durante el entrenamiento, el objetivo es reducir esa perdida. En este proyecto se usa entropia cruzada categorica dispersa, adecuada para clasificacion multiclase.

Optimizador Adam: es el algoritmo que ajusta los pesos. Puede imaginarse como una forma inteligente de corregir los errores del modelo paso a paso, sin cambiar demasiado ni quedarse quieto.

Epoch o epoca: una pasada completa por los datos de entrenamiento. Si se entrenan 50 epocas, el modelo vio el dataset 50 veces, aunque no necesariamente completo si se detiene antes por early stopping.

Batch: grupo pequeno de muestras usado para actualizar los pesos. En vez de aprender con todo el dataset de golpe, el modelo aprende por lotes.

Overfitting o sobreajuste: ocurre cuando el modelo memoriza el dataset pero no generaliza a datos nuevos. Ejemplo: un estudiante que memoriza las respuestas de un examen, pero no entiende el tema. Para reducirlo se usa validacion, dropout, early stopping y prueba con sujetos distintos.

Dropout: tecnica que apaga aleatoriamente algunas neuronas durante el entrenamiento. Obliga a la red a no depender de una sola ruta interna. Es como entrenar a un equipo donde cada jugador debe aprender a responder aunque falte otro.

Normalizacion: transforma los datos para que los canales tengan escalas comparables. Si un canal tiene valores mucho mayores que otro, podria dominar injustamente el entrenamiento.

Class weight: corrige el efecto de clases desbalanceadas. Si hay menos ventanas de pinza que de reposo, el entrenamiento puede darle mas peso a los errores en pinza.

## 5. Ventanas, caracteristicas y senal EMG

La senal EMG es temporal: cambia con el tiempo. Por eso no se clasifica muestra por muestra, sino por ventanas. Una ventana contiene 40 muestras de cada uno de los 4 canales, por lo tanto tiene forma 40 x 4 en los modelos secuenciales.

En los modelos basados en caracteristicas, cada ventana se resume con 7 medidas por canal: media, desviacion estandar, minimo, maximo, media del valor absoluto, RMS y suma de diferencias absolutas. Como son 4 canales, se obtienen 28 caracteristicas por ventana.

Media: valor promedio de la ventana.

Desviacion estandar: que tanto varia la senal.

Minimo y maximo: limites de amplitud en la ventana.

MAV: valor absoluto medio; aproxima la intensidad de activacion muscular.

RMS: raiz media cuadratica; tambien representa energia o potencia de la senal.

Suma de diferencias absolutas: mide que tan rapido cambia la senal.

Ejemplo sencillo: si dos gestos tienen amplitudes parecidas, tal vez se distingan por la variacion o por la forma en que suben y bajan los canales. Por eso no basta con mirar solo el valor maximo.

## 6. Modelo RandomForest baseline

RandomForest no es una red neuronal. Es un modelo clasico de aprendizaje automatico compuesto por muchos arboles de decision. Cada arbol hace preguntas sobre las caracteristicas: por ejemplo, si el RMS del canal 2 es mayor que cierto umbral y la variacion del canal 4 es baja, puede votar por una clase. El bosque combina los votos de muchos arboles para obtener una prediccion mas robusta.

En este proyecto, RandomForest recibe 28 caracteristicas por ventana. No recibe directamente la secuencia 40 x 4, sino un resumen estadistico de esa ventana. Por eso es rapido para validar si las clases son separables.

Ventajas:
- Funciona bien con datasets pequenos o medianos.
- No necesita TensorFlow.
- Es robusto como modelo base y suele entrenarse rapido.
- En la comparativa actual obtuvo 78.61 %, el mejor resultado numerico del conjunto.
- Sirve para responder si las senales contienen informacion suficiente para separar movimientos.

Desventajas:
- No aprende la evolucion temporal fina dentro de la ventana; depende de caracteristicas manuales.
- El modelo guardado puede ser grande para un microcontrolador.
- No es candidato directo para XIAO ESP32S3.
- Puede funcionar muy bien en computador, pero no necesariamente es la mejor opcion embebida.

Como defenderlo: RandomForest se usa como linea base de validacion. Si incluso un modelo clasico logra separar las clases, entonces las senales adquiridas tienen informacion util. No se plantea como solucion final embebida, sino como referencia de escritorio.

## 7. Modelo Tiny MLP TFLite

MLP significa perceptron multicapa. Es una red neuronal de capas densas. En el proyecto, el Tiny MLP recibe las mismas 28 caracteristicas usadas por RandomForest y pasa por esta estructura: entrada de 28 valores, capa densa de 48 neuronas con ReLU, dropout de 0.15, capa densa de 24 neuronas con ReLU y salida de 10 neuronas con softmax.

La palabra Tiny indica que el modelo esta pensado para ser pequeno. Su archivo TFLite pesa aproximadamente 6.8 KB, lo cual lo vuelve muy atractivo para microcontroladores.

Ventajas:
- Es muy pequeno.
- Convierte correctamente a TensorFlow Lite.
- Es viable para XIAO ESP32S3.
- Su inferencia es rapida porque trabaja con 28 caracteristicas, no con toda la secuencia cruda.
- Es facil explicar su estructura: caracteristicas -> capas densas -> probabilidad de gesto.

Desventajas:
- Su exactitud actual es menor: 63.16 %.
- Depende de caracteristicas manuales; si esas caracteristicas no capturan bien la diferencia entre gestos, el modelo no puede inventar informacion nueva.
- Puede confundirse en movimientos parecidos, como cierre general frente a cierre de grupos de dedos.
- Al ser compacto, tiene menos capacidad para modelar patrones complejos.

Ejemplo cotidiano: RandomForest seria como consultar a muchos expertos que votan usando reglas; Tiny MLP seria como una calculadora pequena que aprende a combinar varias pistas numericas para dar una respuesta. Es menos pesado, pero tambien tiene menos memoria y capacidad.

Como defenderlo: Tiny MLP es la ruta minima de despliegue embebido. Aunque no sea el mas preciso, permite demostrar el flujo completo hacia XIAO y sirve como base para optimizar con mas datos o mejores caracteristicas.

## 8. Modelo CNN1D adaptado de iyeleswarapu

CNN significa red neuronal convolucional. En imagenes, una CNN detecta bordes, texturas y formas. En senales EMG, una CNN 1D detecta patrones locales a lo largo del tiempo: picos, cambios de amplitud, transiciones y combinaciones entre canales dentro de una ventana.

En este proyecto, el CNN1D recibe la ventana completa de 40 x 4. No usa caracteristicas manuales. Aprende filtros automaticamente. Su arquitectura adaptada es: Conv1D con 64 filtros y kernel 3, MaxPooling1D, Conv1D con 128 filtros y kernel 3, GlobalAveragePooling1D y salida densa softmax de 10 clases.

Conv1D: aplica pequenos filtros que recorren la senal en el tiempo. Un kernel de 3 observa grupos de 3 muestras consecutivas. Si a 200 Hz cada muestra representa 5 ms, ese filtro mira patrones muy cortos de unos 15 ms.

Filtro: conjunto de pesos que aprende a detectar un patron. Ejemplo: un filtro podria activarse cuando el canal 1 sube mientras el canal 3 se mantiene bajo.

MaxPooling: reduce la longitud temporal conservando activaciones importantes. Es como resumir: de esta zona me quedo con lo mas relevante.

GlobalAveragePooling: promedia las activaciones finales para obtener una representacion compacta antes de clasificar.

Ventajas:
- Aprende caracteristicas automaticamente desde la forma temporal de la senal.
- Convierte correctamente a TFLite.
- El TFLite pesa alrededor de 35.6 KB, todavia razonable para pruebas embebidas.
- Su exactitud actual es 72.66 %, superior al Tiny MLP.
- Es mejor candidato profundo para XIAO porque no usa LSTM.

Desventajas:
- Necesita mas datos que un modelo clasico para generalizar bien.
- Puede no capturar dependencias temporales largas tan bien como una LSTM.
- Consume mas recursos que Tiny MLP.
- La exactitud es menor que RandomForest y CNN-LSTM en la comparativa actual.

Como defenderlo: CNN1D es el balance entre aprendizaje profundo y viabilidad embebida. Aprende patrones temporales sin recurrir a capas recurrentes costosas. Por eso se considera el candidato profundo principal para migrar a XIAO.

## 9. Modelo CNN-LSTM adaptado de ocjorge

CNN-LSTM combina dos ideas: una CNN extrae patrones locales y una LSTM analiza la evolucion temporal de esos patrones. En el proyecto, la entrada tambien es una ventana 40 x 4. La arquitectura adaptada usa Conv1D de 64 filtros con kernel 9, BatchNormalization, MaxPooling, Dropout, otra Conv1D de 128 filtros con kernel 9, BatchNormalization, MaxPooling, Dropout, una LSTM de 96 unidades, otro Dropout y salida softmax.

La CNN funciona como extractor de caracteristicas. Detecta patrones locales: cambios de amplitud, pequenas formas temporales, relaciones entre canales. La LSTM recibe esa representacion y aprende el orden temporal: que ocurre primero, que ocurre despues y como evoluciona la contraccion.

LSTM significa Long Short-Term Memory. Es un tipo de red recurrente disenada para recordar informacion de pasos anteriores. En una senal EMG, esto ayuda porque un gesto no siempre aparece como una foto instantanea; tiene inicio, activacion, sostenimiento y relajacion.

BatchNormalization estabiliza el entrenamiento ajustando las distribuciones internas de activacion. En palabras sencillas, evita que las capas reciban valores demasiado cambiantes durante el aprendizaje.

Ventajas:
- Modela bien datos secuenciales.
- Combina extraccion espacial-temporal de CNN con memoria temporal de LSTM.
- En la comparativa actual obtuvo 78.26 %, muy cerca de RandomForest.
- Es util para demostrar que el componente temporal de la senal aporta informacion.

Desventajas:
- Es mas pesado computacionalmente.
- La conversion a TFLite builtin fallo por operaciones asociadas a LSTM/TensorList.
- No es el primer candidato para XIAO ESP32S3.
- Puede aumentar latencia y consumo de memoria.
- Requiere mas datos y cuidado para no sobreajustar.

Como defenderlo: CNN-LSTM se mantiene como comparador de computador. Sirve para evaluar el beneficio de modelar la temporalidad, pero no bloquea el despliegue embebido porque el proyecto prioriza Tiny MLP y CNN1D para XIAO.

## 10. Modelo Inception-LSTM inspirado en laboratorioAI

El modelo Inception-LSTM parte de una idea llamada bloque Inception. En vez de usar un solo tamano de filtro, procesa la senal con filtros de varios tamanos en paralelo: kernel 1, kernel 3, kernel 5 y una rama con MaxPooling. Luego concatena todo. Esto permite detectar patrones de diferentes escalas temporales.

En el proyecto, el bloque Inception 1D usa varias convoluciones paralelas, normalizacion por lotes y activacion ReLU. Despues se aplican Dropout, otro bloque Inception, MaxPooling, una LSTM de 64 unidades y softmax.

Ejemplo sencillo: si quieres reconocer musica, no basta mirar una sola nota; puedes mirar notas individuales, pequenos grupos y fragmentos mas largos. Inception hace algo parecido: mira la senal con diferentes tamanos de lupa al mismo tiempo.

Ventajas:
- Captura patrones de diferentes duraciones dentro de la misma ventana.
- Combina analisis multiescala con memoria temporal.
- En la comparativa actual obtuvo 74.38 %.
- Tiene respaldo conceptual en trabajos que comparan CNN y CNN-LSTM para reconocimiento de gestos con EMG.

Desventajas:
- Es mas complejo que CNN1D.
- Usa LSTM, por lo que tambien fallo la conversion TFLite builtin.
- Puede ser dificil justificar como modelo embebido inicial.
- La complejidad adicional no supero a CNN-LSTM ni a RandomForest en la comparativa actual.

Como defenderlo: Inception-LSTM se usa como referencia metodologica. Permite comparar si una arquitectura multiescala temporal aporta mejoras. En el estado actual, no se prioriza para XIAO por complejidad y conversion.

## 11. Comparativa actual de modelos

Todos los modelos fueron reentrenados o verificados bajo las mismas condiciones: dataset final de 10 clases, 34 CSV validos, 4 sujetos, 200 Hz, ventana de 40 muestras y stride de 10 muestras.

| Modelo | Exactitud test | Uso practico |
|---|---:|---|
| RandomForest baseline | 78.61 % | Referencia rapida en computador |
| CNN-LSTM ocjorge adaptado | 78.26 % | Comparacion temporal en computador |
| Inception-LSTM laboratorioAI adaptado | 74.38 % | Comparacion temporal en computador |
| CNN1D iyeleswarapu adaptado | 72.66 % | Candidato profundo TFLite |
| Tiny MLP TFLite | 63.16 % | Candidato embebido minimo |

La lectura correcta no es escoger automaticamente el porcentaje mayor. Para la tesis hay dos criterios: rendimiento y viabilidad de despliegue. RandomForest y CNN-LSTM tienen buen rendimiento en computador, pero no son los mas adecuados para la XIAO. CNN1D y Tiny MLP tienen menor exactitud, pero son convertibles a TFLite y por eso son candidatos reales para firmware.

Si te preguntan cual es el mejor, responde con matiz: el mejor en exactitud de prueba es RandomForest; el mejor candidato profundo para XIAO es CNN1D; el modelo minimo embebido es Tiny MLP; los modelos con LSTM son comparadores de computador por su costo y problemas de conversion.

## 12. Por que 200 Hz, ventana 40 y stride 10

El sistema trabaja a 200 Hz para reducir carga de adquisicion, procesamiento e inferencia. A mayor frecuencia, llegan mas muestras por segundo y el microcontrolador tiene menos tiempo para leer ADC, organizar datos, ejecutar inferencia y generar PWM. Para control funcional de una protesis, no se busca reconstruir toda la fisiologia de alta frecuencia de la EMG, sino reconocer patrones suficientemente estables para clasificar gestos.

Con 200 Hz, cada muestra ocurre cada 5 ms. Una ventana de 40 muestras equivale a 200 ms. Ese tiempo es razonable porque una contraccion muscular voluntaria no necesita clasificarse en una sola muestra; se puede reconocer en un fragmento corto. El stride de 10 muestras permite actualizar la prediccion cada 50 ms, lo que da una sensacion cercana a tiempo real.

Trade-off: si la ventana es muy corta, el modelo tiene poca informacion y puede equivocarse. Si es muy larga, la respuesta se vuelve lenta. Si el stride es pequeno, la respuesta se actualiza rapido pero hay mas computo. Si el stride es grande, hay menos carga pero la respuesta se siente mas lenta.

## 13. Como se hace la inferencia en vivo

En inferencia, el sistema no vuelve a entrenar. Usa pesos ya aprendidos. El flujo es: llegan muestras seriales desde la XIAO, se llena un buffer de 40 muestras, se aplica el mismo preprocesamiento usado en entrenamiento y se ejecuta el modelo seleccionado.

Para RandomForest y Tiny MLP, el sistema calcula las 28 caracteristicas por ventana y luego predice. Para CNN1D, CNN-LSTM e Inception-LSTM, el sistema usa la ventana 40 x 4 normalizada. La salida final es una distribucion de probabilidades por clase. El dashboard muestra la clase con mayor probabilidad y las probabilidades principales.

Es fundamental usar el mismo preprocesamiento en entrenamiento e inferencia. Si durante entrenamiento se normalizo con cierta media y desviacion estandar, durante inferencia se deben aplicar esos mismos valores. Cambiar el preprocesamiento equivale a hablarle al modelo en otro idioma.

## 14. Que significa convertir a TensorFlow Lite

TensorFlow/Keras se usa para entrenar en computador. TensorFlow Lite es un formato optimizado para ejecutar modelos en dispositivos con menos recursos. TensorFlow Lite Micro es la variante pensada para microcontroladores.

Convertir un modelo a TFLite no garantiza que funcione en la XIAO. Primero debe convertirse correctamente; luego debe caber en memoria; despues debe ejecutarse con latencia aceptable; y finalmente debe integrarse con el firmware, la adquisicion y la salida PWM.

En el proyecto, Tiny MLP y CNN1D generan archivos TFLite correctamente. CNN-LSTM e Inception-LSTM fallan por operaciones recurrentes asociadas a LSTM/TensorList. Esto no invalida esos modelos; significa que son mas adecuados para comparacion en computador que para el primer despliegue embebido.

## 15. Preguntas probables y respuestas defendibles

Pregunta: Por que no se usan las 14 clases originales?
Respuesta: Porque para control protesico funcional conviene priorizar clases repetibles y utiles. Se redujo a 10 clases que agrupan movimientos relevantes: apertura, cierre, pulgar, indice-medio, anular-menique, pinza y reposo. Esto disminuye carga de entrenamiento y mejora viabilidad con usuarios reales.

Pregunta: Por que comparar varios modelos?
Respuesta: Porque una sola metrica no basta. Se comparan modelos clasicos, compactos y profundos para analizar precision, complejidad, latencia y posibilidad de despliegue embebido.

Pregunta: Si RandomForest tiene mayor exactitud, por que no usarlo en la XIAO?
Respuesta: Porque RandomForest es una buena referencia en PC, pero no es el modelo mas conveniente para microcontrolador por tamano, estructura y despliegue. Para XIAO se priorizan modelos TFLite compactos.

Pregunta: Por que LSTM no va directo a la XIAO?
Respuesta: Porque LSTM requiere mas memoria y operaciones recurrentes. En la conversion actual, TensorFlow Lite builtin no soporta directamente ciertas operaciones TensorList generadas por la LSTM. Por eso se mantiene como comparador de computador.

Pregunta: Que aprende realmente la red?
Respuesta: Aprende relaciones entre canales y patrones temporales asociados a cada gesto. Por ejemplo, una pinza no se define solo por un valor alto, sino por la combinacion de activaciones musculares en varios canales durante un fragmento de tiempo.

Pregunta: Que falta para validar completamente?
Respuesta: Aumentar datos, evaluar por sujeto, medir matriz de confusion final, latencia de adquisicion/inferencia, respuesta PWM y prueba funcional con la protesis.

## 16. Guion corto para explicar en exposicion

Una forma clara de presentarlo seria:

Primero disene un protocolo funcional de 10 clases para controlar movimientos utiles de la protesis. La XIAO ESP32S3 adquiere cuatro canales sEMG a 200 Hz y el computador guarda esos datos como CSV etiquetados. Despues divido la senal en ventanas de 40 muestras, equivalentes a 200 ms, con un desplazamiento de 10 muestras para mantener actualizacion rapida.

Entrene cinco rutas de clasificacion. RandomForest sirve como linea base clasica para comprobar separabilidad de las senales. Tiny MLP es una red pequena que usa caracteristicas estadisticas y se puede exportar a TFLite. CNN1D trabaja directamente con la ventana temporal y aprende filtros de la senal; por eso es el candidato profundo para XIAO. CNN-LSTM e Inception-LSTM se usan como comparadores temporales en computador porque modelan la evolucion de la senal, aunque su conversion embebida no es directa.

Con la comparativa actual, RandomForest obtuvo 78.61 %, CNN-LSTM 78.26 %, Inception-LSTM 74.38 %, CNN1D 72.66 % y Tiny MLP 63.16 %. La decision no se basa solo en exactitud: para el despliegue real se necesita que el modelo sea preciso, pequeno, rapido y compatible con el microcontrolador.

## 17. Glosario rapido

sEMG: electromiografia superficial. Mide actividad electrica muscular mediante electrodos sobre la piel.

ADC: conversor analogico-digital. Convierte la senal analogica del sensor en numeros que lee la XIAO.

Dataset: conjunto de datos usado para entrenar y probar modelos.

Clase: categoria que el modelo debe reconocer, por ejemplo pinza o reposo.

Etiqueta: nombre real asignado a un segmento de senal durante adquisicion.

Ventana: fragmento corto de senal usado como unidad de clasificacion.

Stride: desplazamiento entre una ventana y la siguiente.

Feature o caracteristica: medida extraida de la senal, como RMS o desviacion estandar.

RMS: medida de energia de la senal.

MAV: media del valor absoluto; aproxima intensidad de activacion muscular.

Modelo: funcion entrenada que transforma una entrada en una prediccion.

Entrenamiento: proceso donde el modelo ajusta sus parametros usando datos etiquetados.

Inferencia: uso del modelo ya entrenado para clasificar datos nuevos.

Exactitud: proporcion de predicciones correctas.

Matriz de confusion: tabla que muestra en que clases acierta y en cuales se confunde el modelo.

TFLite: formato liviano de TensorFlow para ejecutar modelos fuera del entorno completo de entrenamiento.

Latencia: tiempo entre adquirir una senal y obtener la respuesta del sistema.

PWM: senal de control usada para posicionar servomotores.

## 18. Fuentes y archivos consultados

Repositorios y referencias externas:
- ocjorge/CNN-LSTM: repositorio de reconocimiento de gestos sEMG con arquitectura hibrida CNN-LSTM sobre NinaPro DB1. URL: https://github.com/ocjorge/CNN-LSTM
- iyeleswarapu/emg-gesture-recognition: proyecto exploratorio de clasificacion de gestos EMG con pipeline de preprocesamiento y CNN-LSTM. URL: https://github.com/iyeleswarapu/emg-gesture-recognition
- laboratorioAI/2023-HGR5-CNN_LSTM: codigo del trabajo sobre influencia de LSTM y postprocesamiento en reconocimiento de gestos con EMG. URL: https://github.com/laboratorioAI/2023-HGR5-CNN_LSTM

Archivos locales del proyecto:
- src/training/train_baseline_rf.py: entrenamiento RandomForest.
- src/training/train_tiny_mlp_tflite.py: entrenamiento Tiny MLP y exportacion TFLite.
- src/training/train_reference_deep_models.py: entrenamiento CNN1D, CNN-LSTM e Inception-LSTM.
- src/training/sequence_dataset.py: carga, segmentacion, ventaneo y normalizacion de secuencias.
- src/common/labels.py: clases finales y alias de etiquetas.
- results/metrics/model_comparison.csv: comparativa actual de modelos.

## 19. Arquitecturas exactas usadas en el proyecto

| Modelo | Entrada | Capas principales | Salida | Compatibilidad |
|---|---|---|---|---|
| RandomForest | 28 caracteristicas por ventana | 250 arboles de decision balanceados | Clase por votacion | PC |
| Tiny MLP | 28 caracteristicas por ventana | Dense 48 ReLU, Dropout 0.15, Dense 24 ReLU | Softmax 10 clases | TFLite OK |
| CNN1D | Ventana 40 x 4 | Conv1D 64 k=3, MaxPool, Conv1D 128 k=3, GlobalAveragePooling | Softmax 10 clases | TFLite OK |
| CNN-LSTM | Ventana 40 x 4 | Conv1D 64 k=9, BN, MaxPool, Dropout, Conv1D 128 k=9, BN, MaxPool, LSTM 96 | Softmax 10 clases | PC; TFLite falla |
| Inception-LSTM | Ventana 40 x 4 | Bloques Inception 1D con filtros k=1, k=3, k=5 y pooling, LSTM 64 | Softmax 10 clases | PC; TFLite falla |

La entrada 40 x 4 significa 40 instantes de tiempo y 4 canales EMG. En los modelos con caracteristicas, esa ventana se transforma antes en 28 valores. En los modelos secuenciales, la red ve la forma temporal completa.

## 20. Diferencia entre entrenamiento, validacion, prueba e inferencia

Entrenamiento: etapa donde el modelo aprende. Se le entregan ventanas EMG con su etiqueta real. El modelo predice, se equivoca, calcula una perdida y ajusta sus pesos. Esta etapa es pesada y se hace en computador.

Validacion: etapa usada durante el entrenamiento para revisar si el modelo esta aprendiendo de forma general o si solo memoriza. En Keras se usa validation_split y early stopping. Si la perdida de validacion deja de mejorar, el entrenamiento se detiene y recupera los mejores pesos.

Prueba o test: etapa final para medir el rendimiento con datos que el modelo no uso directamente para ajustar pesos. La exactitud reportada en la comparativa sale de esta separacion interna.

Inferencia: etapa de uso real. El modelo ya no aprende; solo recibe una ventana nueva y devuelve la clase estimada. En la exposicion es importante separar entrenamiento e inferencia: entrenar es aprender; inferir es aplicar lo aprendido.

## 21. Interpretacion de metricas

Exactitud o accuracy: porcentaje de ventanas clasificadas correctamente. Es facil de entender, pero no siempre cuenta toda la historia. Si una clase tiene muchas mas muestras que otra, un modelo podria tener buena exactitud general y aun asi fallar en clases minoritarias.

Matriz de confusion: muestra en que clases se equivoca el modelo. Por ejemplo, si anular_menique_cierre se confunde con cierre, la matriz permite verlo. Para la tesis, esta grafica sera clave en el capitulo de resultados.

Precision por clase: de todas las veces que el modelo dijo una clase, cuantas eran correctas.

Recall o sensibilidad por clase: de todas las muestras reales de una clase, cuantas encontro el modelo.

F1-score: combina precision y recall. Es util cuando las clases no estan perfectamente balanceadas.

Latencia: tiempo que tarda el sistema desde que adquiere la senal hasta que genera una respuesta. En protesis, una precision alta no basta si la respuesta llega tarde.

## 22. Errores comunes que debes evitar al explicar

No digas: el modelo lee la mente. Mejor: el modelo clasifica patrones electricos musculares asociados a gestos.

No digas: la red sabe que dedo se mueve. Mejor: la red aprende correlaciones entre activaciones musculares y etiquetas de movimiento.

No digas: el modelo de GitHub se uso directamente. Mejor: se adaptaron arquitecturas de referencia y se reentrenaron con el dataset propio.

No digas: el modelo con mayor exactitud siempre es el mejor. Mejor: el mejor depende del objetivo: PC, microcontrolador, latencia, memoria y robustez.

No digas: TFLite significa que ya funciona en la XIAO. Mejor: TFLite es un paso necesario; despues se debe validar memoria, tiempo de inferencia y control PWM en firmware.

## 23. Explicacion para publico no tecnico

El brazalete funciona como un conjunto de microfonos electricos sobre el antebrazo. No escucha sonido, sino actividad muscular. Cuando una persona intenta abrir la mano, cerrar, hacer pinza o mover un grupo de dedos, los musculos no se activan igual. Los sensores capturan esas diferencias.

El computador toma pequenos pedazos de senal, como si fueran pequenos videos de 0.2 segundos. Luego se entrena un modelo para reconocer a que movimiento se parece cada pedazo. Algunos modelos resumen primero la senal en numeros simples, como energia o variacion. Otros modelos miran directamente la forma de la senal en el tiempo.

Despues del entrenamiento, el modelo puede usarse en vivo. Cuando llega una nueva ventana de senal, el modelo devuelve una probabilidad para cada movimiento. Esa clase inferida se puede traducir a una accion de servomotores en la protesis.

## 24. Frase de cierre para exposicion

La contribucion practica de esta etapa no es solo entrenar un modelo, sino construir un flujo completo y comparable: adquisicion propia de sEMG, protocolo funcional de clases, entrenamiento bajo condiciones controladas, comparacion de arquitecturas y seleccion de candidatos viables para computador y para XIAO ESP32S3. Esto permite avanzar hacia una protesis controlada por patrones musculares, con una ruta clara de validacion experimental.
