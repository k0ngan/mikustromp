# -*- coding: utf-8 -*-
"""
build_presentation.py -- Video narrado (~1.5 min) del proyecto Phase Vocoder + Pedal Miku.

Pipeline (igual que el del proyecto previo, adaptado):
  1) Renderiza diapositivas 1920x1080 (matplotlib) que EMBEBEN las figuras de ../figuras.
  2) Narracion en espanol por diapositiva (edge-tts; fallback SAPI/Helena).
  3) Un segmento de video por diapositiva (imagen + narracion) con ffmpeg.
  4) Concatena en phase_vocoder_presentacion.mp4.

Requiere haber corrido antes generar_resultados.py (para que existan las figuras).
"""

import os
import textwrap
import subprocess
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
FIG = os.path.join(PROJ, "figuras")
SLIDES_DIR = os.path.join(HERE, "slides")
AUDIO_DIR = os.path.join(HERE, "audio")
SEG_DIR = os.path.join(HERE, "segmentos")
OUT_MP4 = os.path.join(HERE, "phase_vocoder_presentacion.mp4")

FFMPEG = shutil.which("ffmpeg") or r"C:\ffmpeg\ffmpeg.exe"
FFPROBE = shutil.which("ffprobe") or r"C:\ffmpeg\ffprobe.exe"
for d in (SLIDES_DIR, AUDIO_DIR, SEG_DIR):
    os.makedirs(d, exist_ok=True)

BG = "#0d1b2a"; PANEL = "#16263b"; ACCENT = "#39c6c0"; ACCENT2 = "#ff6fae"
TEXT = "#e8eef5"; MUTED = "#9fb3c8"
plt.rcParams["font.family"] = "DejaVu Sans"
FIG_W, FIG_H, DPI = 16, 9, 120

SLIDES = [
    dict(id=1, kind="titulo",
         titulo="Phase Vocoder",
         subtitulo="Cambiar el tono SIN cambiar la duracion\n(reproduccion de Dolson 1986 + extension de formantes)",
         narracion="Phase vocoder: como cambiar el tono de un sonido sin cambiar su duracion. "
                   "Reproduzco el articulo de Dolson de 1986 y propongo una extension que ademas "
                   "conserva los formantes, todo aplicado a una voz hecha con granos de guitarra."),
    dict(id=2, kind="texto", titulo="El problema",
         bullets=["Subir o bajar el tono de un sonido",
                  "El remuestreo ingenuo cambia la DURACION",
                  "y corre los FORMANTES (efecto ardilla)",
                  "Queremos desacoplar tono, duracion y timbre"],
         narracion="El problema es subir o bajar el tono de un sonido. La via ingenua, el "
                   "remuestreo, cambia la duracion y corre los formantes, el clasico efecto ardilla. "
                   "Queremos desacoplar tono, duracion y timbre."),
    dict(id=3, kind="texto", titulo="El articulo: Dolson 1986",
         bullets=["STFT: magnitud y FASE por banda",
                  "La fase da la frecuencia instantanea",
                  "Reconstruir con otro salto cambia la duracion",
                  "Pitch-shift = time-stretch + remuestreo"],
         narracion="El phase vocoder analiza la senal con la transformada de Fourier de tiempo "
                   "corto y, banda por banda, usa la fase para estimar la frecuencia instantanea. "
                   "Reconstruyendo con un salto distinto se cambia la duracion; y un cambio de tono "
                   "se logra combinando estiramiento temporal y remuestreo."),
    dict(id=4, kind="imagen", titulo="Reproduccion: duracion cambia, tono NO",
         imagen=os.path.join(FIG, "fig01_reproduccion_pv.png"),
         narracion="Aqui esta la reproduccion: al estirar un tono a una vez y media su duracion, "
                   "la senal dura mas pero el tono se mantiene. Esa es la firma del phase vocoder."),
    dict(id=5, kind="imagen", titulo="Extension: preservar los formantes",
         imagen=os.path.join(FIG, "fig03_formantes.png"),
         narracion="Mi extension reimpone la envolvente espectral original usando el cepstrum. "
                   "Asi los armonicos suben de tono pero los formantes quedan fijos: solo el metodo "
                   "phase vocoder con formantes mantiene el timbre."),
    dict(id=6, kind="imagen", titulo="Resultados: duracion y timbre",
         imagen=os.path.join(FIG, "fig02_tono_duracion.png"),
         narracion="En los resultados, el remuestreo deforma la duracion hasta mas menos cincuenta "
                   "por ciento, mientras el phase vocoder la conserva con afinacion sub-audible. La "
                   "extension baja la distorsion del timbre de ocho a menos de dos decibeles. Todo "
                   "con matematica clasica del ramo."),
]


def _new_fig():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor(BG)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")
    return fig, ax


def _header(ax, titulo):
    ax.add_patch(plt.Rectangle((0, 8.1), 16, 0.06, color=ACCENT, lw=0))
    ax.text(0.6, 8.45, titulo, color=TEXT, fontsize=36, fontweight="bold", va="center", ha="left")
    ax.text(15.4, 8.45, "Phase Vocoder  |  PDDI", color=MUTED, fontsize=18, va="center", ha="right")


def _bullets(ax, bullets, x=0.8, y0=6.9, dy=1.05, fs=29, wrap=None):
    y = y0
    for b in bullets:
        lines = textwrap.wrap(b, wrap) if wrap else [b]
        ax.text(x, y, "▸", color=ACCENT, fontsize=fs, va="top", ha="left")
        ax.text(x + 0.55, y, "\n".join(lines), color=TEXT, fontsize=fs, va="top", ha="left", linespacing=1.25)
        y -= dy * (1 + 0.62 * (len(lines) - 1))


def render_titulo(s):
    fig, ax = _new_fig()
    ax.add_patch(plt.Rectangle((0, 3.55), 16, 1.9, color=PANEL, lw=0))
    ax.text(8, 5.6, s["titulo"], color=ACCENT, fontsize=86, fontweight="bold", va="center", ha="center")
    ax.text(8, 4.2, s["subtitulo"], color=TEXT, fontsize=26, va="center", ha="center", linespacing=1.4)
    ax.text(8, 2.0, "Reproduccion de Dolson 1986 + extension de formantes (DSP clasico)",
            color=MUTED, fontsize=20, va="center", ha="center")
    ax.text(8, 1.0, "Francisco Alejandro Pinto Abraham  ·  RUT 21.571.239-7", color=MUTED, fontsize=18,
            va="center", ha="center")
    ax.text(8, 0.5, "Procesamiento Digital de Senales e Imagenes  ·  UTEM 2026-1", color=MUTED,
            fontsize=18, va="center", ha="center")
    return fig


def render_texto(s):
    fig, ax = _new_fig(); _header(ax, s["titulo"]); _bullets(ax, s["bullets"], y0=6.9, dy=1.15, fs=30)
    return fig


def render_imagen(s):
    fig, ax = _new_fig(); _header(ax, s["titulo"])
    try:
        img = plt.imread(s["imagen"])
        sub = fig.add_axes([0.06, 0.08, 0.88, 0.70]); sub.imshow(img); sub.axis("off")
    except Exception as e:
        ax.text(8, 4, "figura no disponible", color=TEXT, ha="center")
        print("  [aviso] no se pudo cargar la figura:", e)
    return fig


RENDERERS = {"titulo": render_titulo, "texto": render_texto, "imagen": render_imagen}


def render_slides():
    print("== Renderizando diapositivas ==")
    paths = []
    for s in SLIDES:
        fig = RENDERERS[s["kind"]](s)
        p = os.path.join(SLIDES_DIR, f"slide_{s['id']:02d}.png")
        fig.savefig(p, dpi=DPI, facecolor=BG); plt.close(fig)
        print("  ok", os.path.basename(p)); paths.append(p)
    return paths


def tts_edge(texto, out_mp3, voice="es-ES-ElviraNeural"):
    cmd = ["python", "-m", "edge_tts", "--voice", voice, "--rate=-4%",
           "--text", texto, "--write-media", out_mp3]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if r.returncode != 0 or not os.path.exists(out_mp3) or os.path.getsize(out_mp3) < 1024:
        raise RuntimeError("edge-tts fallo: " + (r.stderr or "")[:300])
    return out_mp3


def tts_sapi(texto, out_wav):
    ps = ("Add-Type -AssemblyName System.Speech;"
          "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "try { $s.SelectVoice('Microsoft Helena Desktop') } catch {};"
          "$s.Rate = -1;"
          f"$s.SetOutputToWaveFile('{out_wav}');"
          "$s.Speak([Console]::In.ReadToEnd());$s.Dispose()")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       input=texto, capture_output=True, text=True, timeout=90)
    if not os.path.exists(out_wav) or os.path.getsize(out_wav) < 1024:
        raise RuntimeError("SAPI fallo: " + (r.stderr or "")[:300])
    return out_wav


def make_narration():
    print("== Generando narracion (TTS) ==")
    engine = None; audios = []
    for s in SLIDES:
        i = s["id"]; done = None
        if engine in (None, "edge"):
            try:
                mp3 = os.path.join(AUDIO_DIR, f"voz_{i:02d}.mp3")
                tts_edge(s["narracion"], mp3); engine = "edge"; done = mp3
            except Exception as e:
                print(f"  [edge-tts no disponible: {str(e)[:80]}] -> uso SAPI"); engine = "sapi"
        if done is None:
            wav = os.path.join(AUDIO_DIR, f"voz_{i:02d}.wav")
            tts_sapi(s["narracion"], wav); done = wav
        print(f"  ok voz_{i:02d}  ({os.path.basename(done)})"); audios.append(done)
    print("  motor TTS usado:", engine)
    return audios


def duration(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nokey=1:noprint_wrappers=1", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def build_segments(slide_paths, audio_paths, tail=0.6):
    print("== Creando segmentos de video ==")
    segs = []; total = 0.0
    for s, img, aud in zip(SLIDES, slide_paths, audio_paths):
        dur = duration(aud) + tail; total += dur
        seg = os.path.join(SEG_DIR, f"seg_{s['id']:02d}.mp4")
        cmd = [FFMPEG, "-y", "-loop", "1", "-i", img, "-i", aud,
               "-filter_complex", f"[1:a]apad=pad_dur={tail},aresample=44100[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-tune", "stillimage",
               "-pix_fmt", "yuv420p", "-r", "25", "-t", f"{dur:.3f}",
               "-c:a", "aac", "-b:a", "192k", seg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg segmento {s['id']} fallo:\n" + r.stderr[-600:])
        print(f"  ok seg_{s['id']:02d}  {dur:5.1f} s"); segs.append(seg)
    print(f"  duracion total estimada: {total:.1f} s")
    return segs, total


def concat(segs):
    print("== Concatenando video final ==")
    lst = os.path.join(SEG_DIR, "lista.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for s in segs:
            f.write("file '" + s.replace("\\", "/") + "'\n")
    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", OUT_MP4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", lst,
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", OUT_MP4]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("concat fallo:\n" + r.stderr[-600:])
    return OUT_MP4


def main():
    slide_paths = render_slides()
    audio_paths = make_narration()
    segs, total = build_segments(slide_paths, audio_paths)
    out = concat(segs)
    print("\n== LISTO ==")
    print("Video:", out)
    print(f"Duracion final: {duration(out):.1f} s")


if __name__ == "__main__":
    main()
