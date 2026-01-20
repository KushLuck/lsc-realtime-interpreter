# Roadmap — LSC Realtime Interpreter

## V1 (MVP) — señas aisladas (30)
- Captura con webcam + MediaPipe (hands/pose/face)
- Guardado de secuencias de keypoints con etiquetas
- Modelo temporal (baseline GRU o TCN)
- Inferencia en vivo: texto + confianza en pantalla

## V2 — señas continuas (streaming)
- Segmentación inicio/fin (boundary detection) o CTC
- Decodificación streaming (tokens parciales)
- Post-procesado (reglas simples / LM ligero)

## V3 — traducción y salida avanzada
- Glosas → texto natural
- (Opcional) Texto → señas (avatar/animación)
- Optimización y despliegue (web/móvil)
