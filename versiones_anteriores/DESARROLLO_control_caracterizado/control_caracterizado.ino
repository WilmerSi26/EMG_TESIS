#include <ESP32Servo.h>

// --- 1. CONFIGURACIÓN DE PINES ---
const int pinFlexor = 36;   // VP (Sensor que va al músculo de cerrar)
const int pinExtensor = 39; // VN (Sensor que va al músculo de abrir)

// Servos: 0:Pulgar, 1:Indice, 2:Medio, 3:Anular, 4:Meñique
#define NUM_SERVOS 5
Servo servos[NUM_SERVOS];
// Pines de los servos (Ajústalos si cambiaste conexiones)
const int servoPins[NUM_SERVOS] = {23, 22, 21, 17, 16};

// --- 2. VARIABLES DE PROCESAMIENTO (IA) ---
const int VENTANA_TAMANO = 20; // Tamaño de la ventana deslizante
int bufferFlexor[VENTANA_TAMANO];
int bufferExtensor[VENTANA_TAMANO];
int indiceBuffer = 0;

// Filtro de Estabilidad (Debounce de Predicción)
// La IA debe coincidir X veces seguidas para cambiar el movimiento
// Aumenta este número si la mano "tiembla" o cambia muy rápido de opinión
const int COINCIDENCIAS_REQUERIDAS = 6; 
int ultimaPrediccion = 99;
int contadorCoincidencias = 0;
int movimientoActual = 99; // 99 = Reposo

// Prototipo de la función de predicción
int predecirMovimiento(float RMS_Flex, float RMS_Ext, float MAV_Flex, float MAV_Ext, float WL_Flex, float WL_Ext);

void setup() {
  Serial.begin(115200);

  // Inicializar Servos
  for(int i=0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(servoPins[i], 500, 2400);
    servos[i].write(45); // Iniciar en posición relajada (45 grados)
  }

  // Limpiar buffers
  for(int i=0; i<VENTANA_TAMANO; i++) {
    bufferFlexor[i] = 0;
    bufferExtensor[i] = 0;
  }
  
  Serial.println("=== SISTEMA BIÓNICO ONLINE ===");
  Serial.println("Modelo de IA: Arbol de Decision (Depth 8)");
  delay(1000);
}

void loop() {
  // A. LEER SENSORES
  int valFlex = analogRead(pinFlexor);
  int valExt = analogRead(pinExtensor);

  // B. ACTUALIZAR VENTANA DESLIZANTE
  bufferFlexor[indiceBuffer] = valFlex;
  bufferExtensor[indiceBuffer] = valExt;
  indiceBuffer = (indiceBuffer + 1) % VENTANA_TAMANO;

  // C. EXTRAER CARACTERÍSTICAS (FEATURE ENGINEERING)
  // Calculamos RMS, MAV, WL sobre el buffer actual
  float sumCuadradosFlex = 0, sumCuadradosExt = 0;
  float sumAbsFlex = 0, sumAbsExt = 0;
  float sumDiffFlex = 0, sumDiffExt = 0;

  for (int i = 0; i < VENTANA_TAMANO; i++) {
    // RMS & MAV
    sumCuadradosFlex += (float)bufferFlexor[i] * bufferFlexor[i];
    sumCuadradosExt += (float)bufferExtensor[i] * bufferExtensor[i];
    
    sumAbsFlex += abs(bufferFlexor[i]);
    sumAbsExt += abs(bufferExtensor[i]);

    // WL (Waveform Length)
    if (i > 0) {
      // Calculamos la diferencia con el punto anterior en el buffer
      // Nota: Para simplificar en tiempo real ignoramos el salto del índice circular
      sumDiffFlex += abs(bufferFlexor[i] - bufferFlexor[i-1]);
      sumDiffExt += abs(bufferExtensor[i] - bufferExtensor[i-1]);
    }
  }

  // Promedios finales
  float rmsFlex = sqrt(sumCuadradosFlex / VENTANA_TAMANO);
  float rmsExt = sqrt(sumCuadradosExt / VENTANA_TAMANO);
  
  float mavFlex = sumAbsFlex / VENTANA_TAMANO;
  float mavExt = sumAbsExt / VENTANA_TAMANO;
  
  float wlFlex = sumDiffFlex; // WL es acumulativo, no promedio
  float wlExt = sumDiffExt;

  // D. INVOCAR AL CEREBRO (Tu Árbol de Decisión)
  int prediccionInstantanea = predecirMovimiento(rmsFlex, rmsExt, mavFlex, mavExt, wlFlex, wlExt);

  // E. FILTRO DE ESTABILIDAD
  if (prediccionInstantanea == ultimaPrediccion) {
    contadorCoincidencias++;
  } else {
    contadorCoincidencias = 0;
    ultimaPrediccion = prediccionInstantanea;
  }

  if (contadorCoincidencias >= COINCIDENCIAS_REQUERIDAS) {
    if (movimientoActual != prediccionInstantanea) {
       movimientoActual = prediccionInstantanea;
       // Ejecutar movimiento solo cuando cambie el estado confirmado
       ejecutarMovimiento(movimientoActual);
       
       // Debug solo al cambiar
       Serial.print("CAMBIO DETECTADO -> Movimiento: ");
       Serial.println(movimientoActual);
    }
    // Limitamos contador
    contadorCoincidencias = COINCIDENCIAS_REQUERIDAS; 
  }

  delay(10); // Estabilidad del bucle (~100Hz)
}

// --- FUNCIÓN DE CONTROL DE SERVOS ---
void ejecutarMovimiento(int idMovimiento) {
  // 0:Pulgar, 1:Indice, 2:Medio, 3:Anular, 4:Meñique, 5:Puño, 6:Abierta, 99:Reposo
  
  int CERRADO = 180;
  int ABIERTO = 0;
  int REPOSO = 45; 

  switch(idMovimiento) {
    case 0: // PULGAR
      servos[0].write(CERRADO);
      for(int i=1; i<5; i++) servos[i].write(REPOSO);
      break;
      
    case 1: // INDICE
      servos[1].write(CERRADO);
      for(int i=0; i<5; i++) if(i!=1) servos[i].write(REPOSO);
      break;

    case 2: // MEDIO
      servos[2].write(CERRADO);
      for(int i=0; i<5; i++) if(i!=2) servos[i].write(REPOSO);
      break;

    case 3: // ANULAR
      servos[3].write(CERRADO);
      for(int i=0; i<5; i++) if(i!=3) servos[i].write(REPOSO);
      break;

    case 4: // MEÑIQUE
      servos[4].write(CERRADO);
      for(int i=0; i<5; i++) if(i!=4) servos[i].write(REPOSO);
      break;

    case 5: // PUÑO CERRADO
      for(int i=0; i<5; i++) servos[i].write(CERRADO);
      break;

    case 6: // MANO ABIERTA
      for(int i=0; i<5; i++) servos[i].write(ABIERTO);
      break;

    default: // REPOSO
      for(int i=0; i<5; i++) servos[i].write(REPOSO);
      break;
  }
}

// =========================================================
//  TU MODELO DE INTELIGENCIA ARTIFICIAL ENTRENADO
// =========================================================
// Nota: He cambiado los nombres de los argumentos para que coincidan con el cuerpo del código
int predecirMovimiento(float RMS_Flex, float RMS_Ext, float MAV_Flex, float MAV_Ext, float WL_Flex, float WL_Ext) {
  if (MAV_Ext <= 1123.7000) {
    if (WL_Flex <= 628.5000) {
      if (MAV_Ext <= 940.1000) {
        if (RMS_Flex <= 541.7883) {
          if (RMS_Flex <= 492.0504) {
            if (RMS_Ext <= 641.6502) {
              if (WL_Ext <= 266.5000) {
                if (MAV_Ext <= 443.4500) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 1; // DEDO_INDICE
                }
              } else {
                if (MAV_Ext <= 533.1000) {
                  return 0; // DEDO_PULGAR
                } else {
                  return 0; // DEDO_PULGAR
                }
              }
            } else {
              if (RMS_Flex <= 457.5785) {
                return 6; // MANO_ABIERTA
              } else {
                if (RMS_Ext <= 692.1683) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 4; // DEDO_MENIQUE
                }
              }
            }
          } else {
            if (MAV_Ext <= 557.3000) {
              if (RMS_Flex <= 523.9359) {
                if (WL_Flex <= 291.0000) {
                  return 1; // DEDO_INDICE
                } else {
                  return 2; // DEDO_MEDIO
                }
              } else {
                if (WL_Flex <= 291.0000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 2; // DEDO_MEDIO
                }
              }
            } else {
              if (MAV_Flex <= 519.4250) {
                if (RMS_Ext <= 889.2198) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 6; // MANO_ABIERTA
                }
              } else {
                if (WL_Ext <= 653.5000) {
                  return 0; // DEDO_PULGAR
                } else {
                  return 1; // DEDO_INDICE
                }
              }
            }
          }
        } else {
          if (MAV_Ext <= 597.1500) {
            if (WL_Ext <= 322.5000) {
              if (WL_Ext <= 272.0000) {
                if (RMS_Flex <= 557.0914) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 2; // DEDO_MEDIO
                }
              } else {
                if (WL_Flex <= 443.5000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 2; // DEDO_MEDIO
                }
              }
            } else {
              if (RMS_Flex <= 574.2623) {
                if (WL_Flex <= 420.5000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 5; // PUÑO_CERRADO
                }
              } else {
                if (RMS_Ext <= 571.1090) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 2; // DEDO_MEDIO
                }
              }
            }
          } else {
            if (RMS_Flex <= 819.4030) {
              if (MAV_Flex <= 759.1500) {
                if (RMS_Flex <= 601.0453) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 2; // DEDO_MEDIO
                }
              } else {
                if (MAV_Ext <= 755.3500) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 0; // DEDO_PULGAR
                }
              }
            } else {
              if (RMS_Flex <= 875.1879) {
                if (WL_Flex <= 502.5000) {
                  return 1; // DEDO_INDICE
                } else {
                  return 4; // DEDO_MENIQUE
                }
              } else {
                if (MAV_Ext <= 820.2250) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 2; // DEDO_MEDIO
                }
              }
            }
          }
        }
      } else {
        if (RMS_Flex <= 594.9155) {
          if (WL_Flex <= 268.0000) {
            if (MAV_Ext <= 1019.5500) {
              return 0; // DEDO_PULGAR
            } else {
              return 6; // MANO_ABIERTA
            }
          } else {
            if (WL_Flex <= 357.5000) {
              if (WL_Ext <= 1295.0000) {
                if (WL_Flex <= 356.5000) {
                  return 6; // MANO_ABIERTA
                } else {
                  return 4; // DEDO_MENIQUE
                }
              } else {
                return 4; // DEDO_MENIQUE
              }
            } else {
              if (RMS_Flex <= 531.5258) {
                return 6; // MANO_ABIERTA
              } else {
                if (MAV_Flex <= 532.3750) {
                  return 1; // DEDO_INDICE
                } else {
                  return 6; // MANO_ABIERTA
                }
              }
            }
          }
        } else {
          if (MAV_Flex <= 925.7500) {
            if (RMS_Flex <= 808.5469) {
              if (WL_Ext <= 735.0000) {
                if (RMS_Flex <= 784.8902) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 1; // DEDO_INDICE
                }
              } else {
                if (WL_Ext <= 848.0000) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 4; // DEDO_MENIQUE
                }
              }
            } else {
              if (RMS_Flex <= 881.9588) {
                if (WL_Flex <= 548.5000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 4; // DEDO_MENIQUE
                }
              } else {
                if (WL_Flex <= 541.5000) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 4; // DEDO_MENIQUE
                }
              }
            }
          } else {
            if (RMS_Ext <= 971.5505) {
              if (WL_Ext <= 642.0000) {
                return 1; // DEDO_INDICE
              } else {
                if (WL_Ext <= 772.0000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 1; // DEDO_INDICE
                }
              }
            } else {
              if (WL_Ext <= 490.0000) {
                return 4; // DEDO_MENIQUE
              } else {
                if (RMS_Ext <= 975.0023) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            }
          }
        }
      }
    } else {
      if (WL_Flex <= 950.5000) {
        if (MAV_Flex <= 895.1000) {
          if (MAV_Ext <= 924.9250) {
            if (MAV_Ext <= 629.2750) {
              if (RMS_Ext <= 587.0208) {
                if (RMS_Ext <= 473.4939) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                if (WL_Ext <= 522.5000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            } else {
              if (WL_Flex <= 655.5000) {
                if (MAV_Flex <= 795.4000) {
                  return 0; // DEDO_PULGAR
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                if (RMS_Flex <= 832.3861) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 2; // DEDO_MEDIO
                }
              }
            }
          } else {
            if (RMS_Flex <= 750.4963) {
              if (MAV_Flex <= 642.2250) {
                return 6; // MANO_ABIERTA
              } else {
                if (WL_Ext <= 1468.5000) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 6; // MANO_ABIERTA
                }
              }
            } else {
              if (WL_Flex <= 702.0000) {
                if (WL_Ext <= 550.0000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 4; // DEDO_MENIQUE
                }
              } else {
                if (RMS_Flex <= 826.1212) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            }
          }
        } else {
          if (RMS_Ext <= 939.6312) {
            if (RMS_Ext <= 843.6040) {
              if (RMS_Flex <= 919.9766) {
                if (WL_Flex <= 697.0000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 5; // PUÑO_CERRADO
                }
              } else {
                if (MAV_Flex <= 931.6750) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            } else {
              if (WL_Ext <= 766.0000) {
                if (MAV_Flex <= 953.2750) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 5; // PUÑO_CERRADO
                }
              } else {
                if (MAV_Flex <= 1154.1750) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            }
          } else {
            if (WL_Flex <= 677.5000) {
              if (RMS_Flex <= 982.0553) {
                if (WL_Ext <= 467.0000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 4; // DEDO_MENIQUE
                }
              } else {
                if (MAV_Flex <= 1034.6000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            } else {
              if (WL_Ext <= 563.5000) {
                return 3; // DEDO_ANULAR
              } else {
                if (MAV_Ext <= 966.8000) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            }
          }
        }
      } else {
        if (WL_Flex <= 1459.0000) {
          if (WL_Ext <= 584.5000) {
            if (RMS_Ext <= 883.4714) {
              if (MAV_Ext <= 824.8250) {
                if (WL_Ext <= 425.5000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                if (MAV_Flex <= 1049.7000) {
                  return 2; // DEDO_MEDIO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            } else {
              if (RMS_Flex <= 1133.1288) {
                if (MAV_Flex <= 985.6750) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 5; // PUÑO_CERRADO
                }
              } else {
                return 3; // DEDO_ANULAR
              }
            }
          } else {
            if (MAV_Ext <= 955.9250) {
              if (RMS_Ext <= 657.2997) {
                return 3; // DEDO_ANULAR
              } else {
                if (MAV_Flex <= 1177.7750) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            } else {
              if (MAV_Flex <= 1212.6750) {
                if (WL_Flex <= 1028.5000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                return 5; // PUÑO_CERRADO
              }
            }
          }
        } else {
          if (WL_Flex <= 1603.0000) {
            if (WL_Flex <= 1598.5000) {
              if (MAV_Ext <= 912.3250) {
                if (WL_Flex <= 1590.5000) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                if (MAV_Ext <= 928.6000) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            } else {
              return 5; // PUÑO_CERRADO
            }
          } else {
            if (MAV_Ext <= 539.8000) {
              return 5; // PUÑO_CERRADO
            } else {
              return 3; // DEDO_ANULAR
            }
          }
        }
      }
    }
  } else {
    if (MAV_Ext <= 1254.9000) {
      if (MAV_Flex <= 675.3250) {
        return 6; // MANO_ABIERTA
      } else {
        if (MAV_Flex <= 771.1750) {
          return 5; // PUÑO_CERRADO
        } else {
          if (WL_Flex <= 540.5000) {
            if (WL_Flex <= 453.0000) {
              if (WL_Flex <= 439.5000) {
                if (RMS_Ext <= 1137.0826) {
                  return 1; // DEDO_INDICE
                } else {
                  return 3; // DEDO_ANULAR
                }
              } else {
                return 1; // DEDO_INDICE
              }
            } else {
              if (RMS_Ext <= 1251.8912) {
                if (MAV_Ext <= 1127.0750) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 6; // MANO_ABIERTA
                }
              } else {
                if (RMS_Ext <= 1294.8737) {
                  return 1; // DEDO_INDICE
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            }
          } else {
            if (WL_Ext <= 995.0000) {
              if (MAV_Flex <= 969.1000) {
                return 4; // DEDO_MENIQUE
              } else {
                return 3; // DEDO_ANULAR
              }
            } else {
              if (MAV_Flex <= 821.3000) {
                if (WL_Ext <= 1134.5000) {
                  return 4; // DEDO_MENIQUE
                } else {
                  return 6; // MANO_ABIERTA
                }
              } else {
                if (RMS_Ext <= 1262.1661) {
                  return 3; // DEDO_ANULAR
                } else {
                  return 5; // PUÑO_CERRADO
                }
              }
            }
          }
        }
      }
    } else {
      if (WL_Flex <= 871.5000) {
        if (MAV_Ext <= 1373.4750) {
          if (MAV_Ext <= 1372.0750) {
            if (RMS_Ext <= 1385.9398) {
              if (MAV_Flex <= 860.5500) {
                if (RMS_Flex <= 783.9533) {
                  return 6; // MANO_ABIERTA
                } else {
                  return 6; // MANO_ABIERTA
                }
              } else {
                if (WL_Ext <= 1192.5000) {
                  return 6; // MANO_ABIERTA
                } else {
                  return 3; // DEDO_ANULAR
                }
              }
            } else {
              return 1; // DEDO_INDICE
            }
          } else {
            return 3; // DEDO_ANULAR
          }
        } else {
          if (WL_Flex <= 697.5000) {
            if (MAV_Ext <= 1469.9750) {
              if (MAV_Ext <= 1468.8500) {
                if (MAV_Flex <= 803.2750) {
                  return 6; // MANO_ABIERTA
                } else {
                  return 6; // MANO_ABIERTA
                }
              } else {
                return 1; // DEDO_INDICE
              }
            } else {
              if (RMS_Flex <= 764.0017) {
                if (RMS_Flex <= 761.6372) {
                  return 6; // MANO_ABIERTA
                } else {
                  return 5; // PUÑO_CERRADO
                }
              } else {
                return 6; // MANO_ABIERTA
              }
            }
          } else {
            if (WL_Ext <= 1181.5000) {
              return 5; // PUÑO_CERRADO
            } else {
              if (MAV_Ext <= 1969.5750) {
                return 6; // MANO_ABIERTA
              } else {
                if (RMS_Ext <= 2065.1348) {
                  return 5; // PUÑO_CERRADO
                } else {
                  return 6; // MANO_ABIERTA
                }
              }
            }
          }
        }
      } else {
        return 3; // DEDO_ANULAR
      }
    }
  }
}