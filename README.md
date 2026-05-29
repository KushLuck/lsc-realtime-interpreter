# Traductor básico LSC → texto y voz (tiempo real)

Proyecto desarrollado en el marco del **Semillero SEA–UN (Universidad Nacional de Colombia)**.  
Director del semillero: **Prof. Mario Enrique Arrieta-Prieto**. :contentReference[oaicite:2]{index=2}

## Equipo
**Líder:** Juan Fernando Pineda — jpinedap@unal.edu.co :contentReference[oaicite:3]{index=3}  
Juan Ángel Beltrán Gómez — jubeltrango@unal.edu.co :contentReference[oaicite:4]{index=4}  
David Felipe Ángel Herrera — dangelh@unal.edu.co :contentReference[oaicite:5]{index=5}  
Emma Carolina Sarmiento Cabarcas — misarmientoc@unal.edu.co :contentReference[oaicite:6]{index=6}  
Natalia Murcia Ochoa — pmurciao@unal.edu.co :contentReference[oaicite:7]{index=7}  

## Resumen del proyecto
Prototipo que reconoce señas de **Lengua de Señas Colombiana (LSC)** desde cámara en tiempo real y entrega salida en **texto y voz**.  
Objetivo: escalar a un vocabulario de **20 palabras** (saludos, cortesía y expresiones frecuentes). :contentReference[oaicite:8]{index=8}

## Flujo general
1. Captura de muestras por palabra  
2. Normalización de duración  
3. Entrenamiento del modelo  
4. Prueba en vivo (texto + voz) :contentReference[oaicite:9]{index=9}

## Uso rápido (Windows)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.capture.capture_samples
python -m src.train.train
python -m src.infer.live