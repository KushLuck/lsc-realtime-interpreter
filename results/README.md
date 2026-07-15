# Resultados del prototipo LSC

Estos archivos fueron generados con:

```powershell
python -m src.train.plots
```

El script utiliza el modelo, el historial y las 254 muestras del split de
validación existentes. No vuelve a entrenar el modelo.

## Resumen

- Dataset total: 1688 muestras.
- Vocabulario: 20 palabras.
- Validación: 254 muestras (15 % del dataset).
- Accuracy global: 99.21 %.
- Macro precision: 99.06 %.
- Macro recall: 99.40 %.
- Macro F1: 99.20 %.
- Mejor epoch según `val_loss`: 11.
- Mejor `val_loss`: 0.0638.

## Archivos

- `confusion_matrix.png`: palabras reales frente a palabras predichas.
- `training_accuracy.png`: accuracy de entrenamiento y validación por epoch.
- `training_loss.png`: pérdida de entrenamiento y validación por epoch.
- `accuracy_per_class.png`: accuracy de validación para cada palabra.
- `samples_per_class.png`: distribución de las 1688 muestras por clase.
- `pipeline.png`: flujo completo desde la cámara hasta la salida en texto y voz.
- `model_architecture.png`: arquitectura y dimensiones principales de la red GRU.
- `confidence_distribution.png`: confianza de las predicciones y threshold del live.
- `learned_features_pca.png`: proyección 2D de las representaciones internas aprendidas.
- `metrics.json`: métricas completas y reporte de clasificación.
- `metrics_per_class.csv`: resumen por palabra para Excel u otras herramientas.

## Lectura para la presentación

El modelo clasificó correctamente 252 de 254 muestras de validación. Las dos
confusiones observadas fueron una muestra de `gracias` predicha como
`buenos_dias` y una de `bien` predicha como `ayudar`.

Estos valores representan una validación interna del prototipo. La evaluación
con más participantes y sesiones puede presentarse como trabajo futuro.

Para una exposición breve se recomienda priorizar `pipeline.png`,
`model_architecture.png`, `training_accuracy.png`, `confusion_matrix.png` y
`confidence_distribution.png`. El PCA es una visualización exploratoria: ayuda
a mostrar la separación entre clases, pero no reemplaza las métricas de
evaluación.
