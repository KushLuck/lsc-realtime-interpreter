# src/infer/tts_gtts.py
import os
import tempfile
import threading
import queue
from time import sleep

from gtts import gTTS
import pygame


class GTTSWorker:
    """
    Worker TTS asíncrono usando gTTS + pygame.mixer.

    Correcciones aplicadas:
    - pygame.mixer.pre_init() antes de init() para reducir latencia de audio.
    - Se inicializa solo pygame.mixer (no pygame completo) para evitar
      fallos en entornos sin display (headless / Windows sin GUI).
    - Verificación explícita de que mixer se inicializó correctamente.
    - Limpieza de archivo temporal más robusta (espera a que mixer suelte el archivo).
    - Manejo de cola mejorado: evita que texto vacío entre al worker.
    - close() verifica que el hilo esté vivo antes de join().
    """

    def __init__(self, lang: str = "es", debug: bool = False):
        self.lang = lang
        self.debug = debug
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self._stop = threading.Event()

        # ✅ pre_init ANTES de init para reducir latencia de audio
        # buffer=512 → latencia baja; frequency=22050 suficiente para voz
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)

        # ✅ Solo inicializamos el subsistema de audio, no pygame completo.
        # Esto evita el error "No video mode has been set" en entornos headless.
        pygame.mixer.init()

        if not pygame.mixer.get_init():
            raise RuntimeError(
                "No se pudo inicializar pygame.mixer. "
                "Verifica que tengas un dispositivo de audio disponible."
            )

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    # Hilo worker
    # ------------------------------------------------------------------

    def _run(self):
        while not self._stop.is_set():
            text = None
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            # Señal de cierre
            if text is None:
                break

            text = str(text).strip()
            if not text:
                continue

            # Archivo temporal para el mp3 generado
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            try:
                if self.debug:
                    print(f"[gTTS] generating: {text!r}")

                tts_obj = gTTS(text=text, lang=self.lang)
                tts_obj.save(path)

                # Detiene y descarga audio previo antes de cargar el nuevo
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()

                # ✅ unload() antes de load() garantiza que el archivo anterior
                # ya no esté bloqueado por el mixer (crítico en Windows)
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

                pygame.mixer.music.load(path)
                pygame.mixer.music.play()

                # Espera activa sin bloquear el hilo principal
                while pygame.mixer.music.get_busy() and not self._stop.is_set():
                    sleep(0.05)

            except OSError as e:
                # Error de red (gTTS requiere internet) o disco lleno
                print(f"[gTTS ERROR - red/disco] {e}")
            except Exception as e:
                print(f"[gTTS ERROR] {e}")
            finally:
                # ✅ Descargar mixer antes de borrar el archivo temporal.
                # En Windows, si el archivo está en uso por el mixer al momento
                # de os.remove(), lanza PermissionError silencioso.
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass

                # Pequeña espera para asegurar que el SO libere el handle
                sleep(0.05)

                try:
                    os.remove(path)
                except OSError:
                    pass  # Si falla (raro), no es crítico; el SO lo limpiará

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def speak(self, text: str):
        """
        Encola texto para síntesis.
        Descarta entradas anteriores pendientes para hablar siempre
        lo más reciente (evita backlog de señas detectadas).
        """
        text = str(text).strip()
        if not text:
            return

        # Vacía la cola: solo importa la última seña
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        self.q.put(text)

    def close(self):
        """Detiene el worker y libera recursos de audio."""
        self._stop.set()
        self.q.put(None)  # Desbloquea q.get() si está esperando

        # ✅ Solo hace join si el hilo está vivo
        if self.thread.is_alive():
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass

        # Libera subsistema de audio
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass