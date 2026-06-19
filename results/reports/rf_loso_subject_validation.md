# Validacion RandomForest por sujeto

Esta validacion usa Leave-One-Subject-Out: en cada pliegue se entrena con tres sujetos y se evalua con el sujeto restante. Sirve para medir generalizacion inter-sujeto, que es mas exigente que una particion aleatoria de ventanas.

- Dataset: `data\final_10clases_valid`
- Sujetos: S001, S002, S003, S004
- Ventanas: 48897
- Ventana temporal: 40 muestras
- Salto entre ventanas: 10 muestras
- Clases: anular_menique_apertura, anular_menique_cierre, apertura, cierre, indice_medio_apertura, indice_medio_cierre, pinza, pulgar_apertura, pulgar_cierre, reposo
- Exactitud media por sujeto: 0.3111

| Sujeto evaluado | Ventanas test | Exactitud |
|---|---:|---:|
| S001 | 12977 | 0.1737 |
| S002 | 7189 | 0.1626 |
| S003 | 14376 | 0.5120 |
| S004 | 14355 | 0.3962 |

## Interpretacion

Si esta metrica es menor que la validacion aleatoria, no significa que el sistema falle; significa que las senales sEMG cambian entre personas. Para control protesico real se puede justificar calibracion por usuario y reportar tambien validacion intra-sujeto por trials.
