# Dataset colaborativo (keypoints + metadata)

Este repositorio versiona **solo**:
- `data/keypoints_v1/` → archivos `.npy` (secuencias de keypoints por muestra)
- `data/metadata/` → archivos `.json` (metadatos por muestra)
- `data/README.md` → estas instrucciones

**No se suben** videos ni frames crudos (son muy pesados):
- `data/raw_frames/`
- `data/raw_videos/`

La idea es que el equipo capture datos livianos y los suba al repositorio para entrenar un modelo común.

---

## 1) Primera vez (Windows) – clonar e instalar

### 1.1 Clonar el repositorio
```bash
git clone <URL_DEL_REPO_PRIVADO>
cd lsc-realtime-interpreter
```

### 1.2 Crear y activar entorno virtual (Python 3.11 recomendado)

En PowerShell (Windows):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 1.3 Verificación rápida

```powershell
python -c "import cv2, numpy as np; import mediapipe as mp; import tensorflow as tf; print('OK', np.__version__, cv2.__version__, mp.__version__, tf.__version__)"
```

---

## 2) Antes de capturar: palabras y protocolo

### 2.1 Verifica la palabra en `models/words.json`

La palabra que captures debe existir **exactamente igual** en `models/words.json` (mismo nombre y formato).

### 2.2 Protocolo de captura (para consistencia)

* Misma distancia a cámara (torso y manos visibles).
* Buena luz frontal (evitar contraluz).
* Fondo relativamente estable (no tiene que ser perfecto).
* Secuencia recomendada por muestra:

  1. manos abajo (inicio claro)
  2. hacer la seña completa
  3. pausa corta (0.5–1 s)
  4. manos abajo (cierre claro)

Sugerencia: por palabra, mínimo **40** muestras por persona (ideal 60–100 si hay tiempo).

---

## 3) Captura de datos (por palabra)

### 3.1 Ejecutar captura

```powershell
python -m src.capture.capture_samples
```

En `src/capture/capture_samples.py` se define la palabra que se está capturando (ej.: `"hola"`, `"adios"`).
Captura varias repeticiones y cierra con **Q**.

### 3.2 Archivos generados

Por cada muestra se generan:

* `data/keypoints_v1/<palabra>/<sample_id>.npy`
* `data/metadata/<palabra>/<sample_id>.json`

---

## 4) Subir datos al repositorio (siempre por rama + Pull Request)

### 4.1 Actualizar repo antes de grabar

```powershell
git pull
```

### 4.2 Crear una rama para tus datos

Ejemplo:

```powershell
git checkout -b data/<tu-nombre>-batch1
```

### 4.3 Añadir y commitear SOLO lo necesario

```powershell
git status
git add data/keypoints_v1 data/metadata
git commit -m "data: <tu-nombre> batch1 (palabras: X, Y, Z)"
```

### 4.4 Subir la rama

```powershell
git push -u origin data/<tu-nombre>-batch1
```

Luego crea un Pull Request hacia `main`.

---

## 5) Entrenar y probar (local)

Entrenamiento:

```powershell
python -m src.train.train
```

Prueba en vivo:

```powershell
python -m src.infer.live
```

---

## 6) Reglas rápidas

* No subir `.venv/`
* No subir `data/raw_frames/` ni `data/raw_videos/`
* Subir datos siempre por rama + Pull Request
