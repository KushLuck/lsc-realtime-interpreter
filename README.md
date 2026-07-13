# Traductor básico LSC → texto y voz (tiempo real)

Proyecto desarrollado en el marco del **Semillero SEA–UN (Universidad Nacional de Colombia)**.  
Director del semillero: **Prof. Mario Enrique Arrieta-Prieto**.

## Equipo
**Líder:** Juan Fernando Pineda — jpinedap@unal.edu.co  
Juan Ángel Beltrán Gómez — jubeltrango@unal.edu.co  
David Felipe Ángel Herrera — dangelh@unal.edu.co  
Emma Carolina Sarmiento Cabarcas — misarmientoc@unal.edu.co  
Natalia Murcia Ochoa — pmurciao@unal.edu.co  

## Resumen del proyecto
Prototipo que reconoce señas de **Lengua de Señas Colombiana (LSC)** desde cámara en tiempo real y entrega salida en **texto y voz**.  
Objetivo: escalar a un vocabulario de **20 palabras** (saludos, cortesía y expresiones frecuentes).

## Flujo general
1. Captura de muestras por palabra  
2. Normalización de duración  
3. Entrenamiento del modelo  
4. Prueba en vivo (texto + voz)

## Reglas del repositorio
- **No subir datos**: la carpeta `data/` (muestras, keypoints, videos, metadata) no se versiona por privacidad y tamaño.
- **No subir entornos**: no subir `.venv/`, `venv/`, `__pycache__/` ni archivos temporales.
- **Cambios por ramas**: cada tarea se trabaja en una rama (`feature/...`, `fix/...`, `docs/...`) y se integra por Pull Request.
- **Evitar push directo a main**: lo ideal es que `main` quede protegido y solo se actualice por PR.
- **Commits claros**: mensajes cortos y descriptivos (ej.: `fix: threshold de live`, `docs: actualizar README`, `feat: captura para nuevas palabras`).

## Uso rápido (Windows)

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.capture.capture_samples
python -m src.train.train
python -m src.infer.live
```
```bash
python -m tools.clear_samples --list  
```
La captura de muestras, la inferencia y la prueba de FPS usan 1280x720.
