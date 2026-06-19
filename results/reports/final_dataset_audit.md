# Auditoria del dataset funcional

Fecha de generacion: 2026-06-19T01:12:02

- Archivos revisados: 37
- Archivos validos: 34
- Archivos excluidos: 3
- Sujetos validos: S001, S002, S003, S004
- Filas validas aproximadas: 545012

## Criterio

Un CSV se marca como valido si supera el minimo de filas, contiene los cuatro canales EMG, conserva identificador de sujeto, reporta la frecuencia esperada y contiene las 10 clases funcionales normalizadas.

## Archivos excluidos

| Archivo | Filas | Sujeto | Motivo |
|---|---:|---|---|
| 20260618_230213_S001_SES20260618230213_1rep_funcional_10clases_4ch_200hz.csv | 1162 | S001 | short_file:1162<12000;missing_labels:anular_menique_apertura|anular_menique_cierre|apertura|cierre|indice_medio_apertura|indice_medio_cierre|pinza|pulgar_apertura|pulgar_cierre |
| 20260619_004200_S001_trial-1_funcional_10clases_4ch_200hz.csv | 432 | S001 | short_file:432<12000;missing_labels:anular_menique_apertura|anular_menique_cierre|apertura|cierre|indice_medio_apertura|indice_medio_cierre|pinza|pulgar_apertura|pulgar_cierre|reposo |
| 20260619_004348_S001_trial-8_funcional_10clases_4ch_200hz.csv | 1179 | S001 | short_file:1179<12000;missing_labels:anular_menique_apertura|anular_menique_cierre|apertura|cierre|indice_medio_apertura|indice_medio_cierre|pinza|pulgar_apertura|pulgar_cierre |
