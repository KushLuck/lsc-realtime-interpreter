# src/infer/tts_gtts.py
import os
import tempfile
import threading
import queue
from time import sleep

from gtts import gTTS
import pygame


class GTTSWorker:
    def __init__(self, lang: str = "es", debug: bool = False):
        self.lang = lang
        self.debug = debug
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self._stop = threading.Event()

        pygame.init()
        pygame.mixer.init()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self._stop.is_set():
            text = None
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                break

            text = str(text).strip()
            if not text:
                continue

            # archivo temporal (evita pisarse)
            fd, path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            try:
                if self.debug:
                    print(f"[gTTS] generating: {text}")

                tts = gTTS(text=text, lang=self.lang)
                tts.save(path)

                # corta audio anterior (si lo había)
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

                pygame.mixer.music.load(path)
                pygame.mixer.music.play()

                # espera (sin congelar el hilo principal)
                while pygame.mixer.music.get_busy() and not self._stop.is_set():
                    sleep(0.05)

            except Exception as e:
                print(f"[gTTS ERROR] {e}")
            finally:
                # limpia archivo temporal
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    pass
                try:
                    os.remove(path)
                except Exception:
                    pass

    def speak(self, text: str):
        text = str(text).strip()
        if not text:
            return

        # evita backlog: deja solo lo último
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        self.q.put(text)

    def close(self):
        self._stop.set()
        self.q.put(None)
        try:
            self.thread.join(timeout=2.0)
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass
