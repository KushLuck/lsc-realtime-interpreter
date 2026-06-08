# tools/clear_samples.py
"""
Herramienta interactiva para eliminar muestras capturadas de LSC.

Modos de uso
------------
1. Eliminar TODO el dataset (reset completo):
       python -m tools.clear_samples --all

2. Eliminar todas las muestras de una palabra específica:
       python -m tools.clear_samples --word adios

3. Eliminar muestras individuales de una palabra (selección interactiva):
       python -m tools.clear_samples --word adios --pick

4. Listar muestras sin eliminar nada:
       python -m tools.clear_samples --list
       python -m tools.clear_samples --list --word adios

Estructura esperada en disco
-----------------------------
data/
  keypoints_v1/<word_id>/<sample_id>.npy
  metadata/<word_id>/<sample_id>.json
  raw_frames/<word_id>/<sample_id>/          ← opcional, solo si save_debug_frames=True
"""

import os
import sys
import glob
import json
import shutil
import argparse
from datetime import datetime
from typing import Optional


# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
DATA_ROOT = "data"
KP_ROOT   = os.path.join(DATA_ROOT, "keypoints_v1")
MD_ROOT   = os.path.join(DATA_ROOT, "metadata")
RF_ROOT   = os.path.join(DATA_ROOT, "raw_frames")


# ------------------------------------------------------------------
# Helpers de consola
# ------------------------------------------------------------------
def _confirm(msg: str) -> bool:
    """Pide confirmación explícita al usuario. Solo acepta 'si' o 'no'."""
    while True:
        resp = input(f"\n⚠️  {msg} [si/no]: ").strip().lower()
        if resp in ("si", "sí", "s"):
            return True
        if resp in ("no", "n"):
            return False
        print("   Responde 'si' o 'no'.")


def _fmt_sample(sample_id: str) -> str:
    """Convierte el timestamp en formato legible. ej: 250612143022123456 → 2025-06-12 14:30:22"""
    try:
        dt = datetime.strptime(sample_id[:12], "%y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return sample_id


# ------------------------------------------------------------------
# Lectura del estado actual
# ------------------------------------------------------------------
def get_words() -> list[str]:
    """Devuelve las palabras que tienen al menos un .npy en keypoints_v1."""
    if not os.path.isdir(KP_ROOT):
        return []
    return sorted(
        w for w in os.listdir(KP_ROOT)
        if os.path.isdir(os.path.join(KP_ROOT, w))
        and len(glob.glob(os.path.join(KP_ROOT, w, "*.npy"))) > 0
    )


def get_samples(word_id: str) -> list[str]:
    """Devuelve los sample_ids (.npy sin extensión) de una palabra, ordenados por fecha."""
    pattern = os.path.join(KP_ROOT, word_id, "*.npy")
    files = sorted(glob.glob(pattern))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def count_summary() -> dict[str, int]:
    """Devuelve {word_id: n_muestras} para todas las palabras."""
    return {w: len(get_samples(w)) for w in get_words()}


# ------------------------------------------------------------------
# Eliminación
# ------------------------------------------------------------------
def delete_sample(word_id: str, sample_id: str):
    """Elimina los archivos de una muestra individual (npy + json + raw_frames)."""
    removed = []

    npy_path = os.path.join(KP_ROOT, word_id, f"{sample_id}.npy")
    if os.path.exists(npy_path):
        os.remove(npy_path)
        removed.append(npy_path)

    json_path = os.path.join(MD_ROOT, word_id, f"{sample_id}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
        removed.append(json_path)

    rf_path = os.path.join(RF_ROOT, word_id, sample_id)
    if os.path.isdir(rf_path):
        shutil.rmtree(rf_path)
        removed.append(rf_path + "/")

    return removed


def delete_word(word_id: str) -> int:
    """Elimina todas las muestras de una palabra. Devuelve el número eliminado."""
    samples = get_samples(word_id)
    for s in samples:
        delete_sample(word_id, s)

    # Limpia carpetas vacías
    for root in [KP_ROOT, MD_ROOT, RF_ROOT]:
        folder = os.path.join(root, word_id)
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)

    return len(samples)


def delete_all() -> dict[str, int]:
    """Elimina todas las muestras de todas las palabras."""
    words = get_words()
    result = {}
    for w in words:
        result[w] = delete_word(w)
    return result


# ------------------------------------------------------------------
# Modos de operación
# ------------------------------------------------------------------
def cmd_list(word_id: Optional[str]):
    """Lista el estado actual del dataset."""
    summary = count_summary()

    if not summary:
        print("\n📂 No hay muestras capturadas en data/keypoints_v1/")
        return

    if word_id:
        if word_id not in summary:
            print(f"\n❌ La palabra '{word_id}' no tiene muestras capturadas.")
            return
        samples = get_samples(word_id)
        print(f"\n📋 Muestras de '{word_id}' ({len(samples)} total):\n")
        for i, s in enumerate(samples, 1):
            print(f"   {i:>3}. {s}  ({_fmt_sample(s)})")
    else:
        total = sum(summary.values())
        print(f"\n📋 Dataset actual — {len(summary)} palabras, {total} muestras totales:\n")
        for w, n in sorted(summary.items()):
            bar = "█" * min(n, 40)
            print(f"   {w:<20} {n:>3} muestras  {bar}")
        print()


def cmd_all():
    """Modo --all: elimina todo el dataset."""
    summary = count_summary()

    if not summary:
        print("\n📂 No hay muestras que eliminar.")
        return

    total = sum(summary.values())
    print(f"\n📋 Se eliminarán {total} muestras de {len(summary)} palabras:")
    for w, n in sorted(summary.items()):
        print(f"   • {w:<20} {n} muestras")

    if not _confirm(f"¿Confirmas eliminar LAS {total} MUESTRAS? Esta acción no se puede deshacer."):
        print("\n🚫 Operación cancelada.\n")
        return

    result = delete_all()
    print()
    for w, n in result.items():
        print(f"   🗑️  {w}: {n} muestras eliminadas")
    print(f"\n✅ Listo. Dataset reseteado ({sum(result.values())} muestras eliminadas).\n")


def cmd_word(word_id: str):
    """Modo --word sin --pick: elimina todas las muestras de una palabra."""
    samples = get_samples(word_id)

    if not samples:
        print(f"\n❌ La palabra '{word_id}' no tiene muestras capturadas.")
        return

    print(f"\n📋 Se eliminarán {len(samples)} muestras de '{word_id}'.")

    if not _confirm(f"¿Confirmas eliminar TODAS las muestras de '{word_id}'?"):
        print("\n🚫 Operación cancelada.\n")
        return

    n = delete_word(word_id)
    print(f"\n✅ {n} muestras de '{word_id}' eliminadas.\n")


def cmd_pick(word_id: str):
    """Modo --word + --pick: selección interactiva de muestras a eliminar."""
    samples = get_samples(word_id)

    if not samples:
        print(f"\n❌ La palabra '{word_id}' no tiene muestras capturadas.")
        return

    print(f"\n📋 Muestras de '{word_id}' ({len(samples)} total):\n")
    for i, s in enumerate(samples, 1):
        print(f"   {i:>3}. {s}  ({_fmt_sample(s)})")

    print(
        "\nIngresa los números a eliminar separados por coma, "
        "un rango (ej: 2-5), o combinación (ej: 1,3,5-8)."
    )
    print("   'all'  → eliminar todas")
    print("   'q'    → cancelar\n")

    raw = input("   Selección: ").strip().lower()

    if raw in ("q", ""):
        print("\n🚫 Operación cancelada.\n")
        return

    if raw == "all":
        selected = list(range(len(samples)))
    else:
        selected = set()
        for part in raw.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    a, b = part.split("-")
                    selected.update(range(int(a) - 1, int(b)))
                except ValueError:
                    print(f"   ⚠️  Rango inválido ignorado: '{part}'")
            else:
                try:
                    selected.add(int(part) - 1)
                except ValueError:
                    print(f"   ⚠️  Valor inválido ignorado: '{part}'")
        selected = sorted(selected)

    # Filtrar índices fuera de rango
    selected = [i for i in selected if 0 <= i < len(samples)]

    if not selected:
        print("\n⚠️  No se seleccionó ninguna muestra válida.\n")
        return

    print(f"\n   Se eliminarán {len(selected)} muestra(s):")
    for i in selected:
        print(f"     • [{i+1}] {samples[i]}  ({_fmt_sample(samples[i])})")

    if not _confirm(f"¿Confirmas eliminar estas {len(selected)} muestra(s)?"):
        print("\n🚫 Operación cancelada.\n")
        return

    for i in selected:
        delete_sample(word_id, samples[i])
        print(f"   🗑️  Eliminada: {samples[i]}")

    remaining = len(get_samples(word_id))
    print(f"\n✅ {len(selected)} muestra(s) eliminada(s). Quedan {remaining} para '{word_id}'.\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    # ✅ global debe declararse ANTES de cualquier uso de las variables
    global DATA_ROOT, KP_ROOT, MD_ROOT, RF_ROOT

    parser = argparse.ArgumentParser(
        description="Elimina muestras capturadas de LSC de forma segura.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Elimina todo el dataset (reset completo).",
    )
    parser.add_argument(
        "--word",
        type=str,
        metavar="WORD_ID",
        help="Palabra objetivo (ej: adios, gracias).",
    )
    parser.add_argument(
        "--pick",
        action="store_true",
        help="Selección interactiva de muestras individuales (requiere --word).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista muestras sin eliminar nada.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DATA_ROOT,
        help=f"Ruta raíz del dataset (default: {DATA_ROOT}).",
    )

    args = parser.parse_args()

    # Permitir data-root personalizado
    DATA_ROOT = args.data_root
    KP_ROOT   = os.path.join(DATA_ROOT, "keypoints_v1")
    MD_ROOT   = os.path.join(DATA_ROOT, "metadata")
    RF_ROOT   = os.path.join(DATA_ROOT, "raw_frames")

    # Validaciones de argumentos
    if args.pick and not args.word:
        parser.error("--pick requiere --word.")

    if args.all and args.word:
        parser.error("--all y --word son mutuamente excluyentes.")

    # Despacho
    if args.list:
        cmd_list(args.word)
    elif args.all:
        cmd_all()
    elif args.word and args.pick:
        cmd_pick(args.word)
    elif args.word:
        cmd_word(args.word)
    else:
        # Sin argumentos: mostrar resumen y ayuda
        cmd_list(None)
        print("Usa --help para ver todos los modos de uso.\n")


if __name__ == "__main__":
    main()