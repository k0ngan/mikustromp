# -*- coding: utf-8 -*-
"""
construir_stomp.py -- Arma `miku_stomp_colab.ipynb`, un cuaderno AUTOCONTENIDO para Google Colab
(o local) que implementa el "Miku Stomp digital" en DSP puro.

Embebe el núcleo DSP de `vocoder.py` y `stomp.py` en celdas (fuente única; en Colab no hace falta
subir esos archivos). Las celdas específicas de Colab van guardadas por `IN_COLAB`, de modo que el
cuaderno también se ejecuta local de punta a punta usando respaldos (guitarra de GitHub + voz
sintética), sin intervención manual.
"""

import os
import base64
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# Voz real incluida (Vocaloid 4 'CyberDiva', voz femenina cantada) embebida en el cuaderno como
# base64 para que funcione en Colab sin subir nada. Uso académico, declarado.
_voice_path = os.path.join(HERE, 'voces', 'miku_voice.wav')
VOICE_B64 = base64.b64encode(open(_voice_path, 'rb').read()).decode('ascii') if os.path.exists(_voice_path) else ''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


# --- núcleo DSP embebido (paridad con vocoder.py + stomp.py) ---------------------------
vocoder_src = open(os.path.join(HERE, 'vocoder.py'), encoding='utf-8').read()

stomp_src = open(os.path.join(HERE, 'stomp.py'), encoding='utf-8').read()
# transformar stomp.py para que use los nombres ya definidos por vocoder (sin el alias V)
stomp_src = stomp_src.replace('import vocoder as V\n', '')
stomp_src = stomp_src.replace('V.', '')
stomp_src = stomp_src.replace('SR = SR', 'SR = SR  # ya definido por el núcleo vocoder')

cells = []

cells.append(md(r"""# Miku Stomp digital — replicar el pedal Korg en DSP puro

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1**
Francisco Alejandro Pinto Abraham · RUT 21.571.239-7

El pedal **Korg Miku Stomp** convierte lo que tocas en una "voz de Hatsune Miku". Aquí lo replicamos
en Python con el mismo flujo —**audio de guitarra → detectar la nota → afinar una voz a esa nota →
audio de salida**— pero **mejor que el pedal**: el pedal mueve los *formantes* al cambiar el tono
(efecto "ardilla"); nosotros usamos un **phase vocoder con preservación de formantes** (de nuestro
proyecto del artículo de Dolson 1986), así la voz mantiene su timbre al seguir la melodía.

**Regla del ramo:** todo es DSP clásico — detección de tono **pYIN** (YIN probabilístico, *no* es una
red neuronal), **STFT**, **phase vocoder**, **formantes por cepstrum**, **overlap-add** y **mezcla**.
Sin deep learning / IA generativa (RVC, Basic Pitch, etc. quedan solo como *trabajo futuro*).

> En Colab: **Entorno de ejecución → Ejecutar todo**. Trae **incluida una voz real de Vocaloid 4**
> (CyberDiva), así que suena a voz de verdad sin subir nada. Puedes subir tu propia guitarra y/o tu
> propia voz si quieres; si no, usa una guitarra de ejemplo y la voz incluida, y corre de principio a fin.
"""))

cells.append(md("## 0. Dependencias"))
cells.append(code(r"""# En Colab instala lo necesario; local: si ya está, no hace nada.
try:
    import google.colab  # noqa
    !pip install -q librosa soundfile numpy scipy matplotlib
except Exception:
    pass
print("Dependencias listas.")
"""))

cells.append(code(r"""import os, io, urllib.request, warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
warnings.filterwarnings("ignore")
try:
    import google.colab  # noqa
    IN_COLAB = True
except Exception:
    IN_COLAB = False
print("IN_COLAB =", IN_COLAB)
"""))

cells.append(md("## 1. Núcleo DSP (phase vocoder + formantes)\n"
                "Mismas funciones validadas en `vocoder.py` (Dolson 1986 + extensión de formantes)."))
cells.append(code(vocoder_src))

cells.append(md("## 2. Núcleo del pedal (detección de notas + síntesis)\n"
                "Funciones de `stomp.py`: `track_f0_pyin`, `segment_notes`, `prep_voice_grain`, "
                "`miku_stomp`."))
cells.append(code(stomp_src))

cells.append(md("## 3. Entrada: la guitarra\n"
                "En Colab puedes subir tu propio audio; si no, se usa una guitarra de ejemplo."))
cells.append(code(r"""GUITAR_URL = "https://raw.githubusercontent.com/k0ngan/audio/main/guitarra.mp3"
SECONDS = 5.0

def _load_any(path, seconds):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return ensure_mono_float(y)[:int(seconds * SR)]

def load_guitar(seconds=SECONDS):
    # 1) subir en Colab
    if IN_COLAB:
        try:
            from google.colab import files
            print("Sube tu guitarra (.wav/.mp3); o pulsa Cancelar para usar la de ejemplo.")
            up = files.upload()
            if up:
                return _load_any(list(up.keys())[0], seconds), "subida"
        except Exception as e:
            print("  (sin upload:", str(e)[:60], ")")
    # 2) copia local / descarga
    for p in ("guitarra.mp3", "data/guitarra.mp3"):
        if os.path.exists(p):
            return _load_any(p, seconds), "local: " + p
    try:
        urllib.request.urlretrieve(GUITAR_URL, "guitarra.mp3")
        return _load_any("guitarra.mp3", seconds), "GitHub"
    except Exception as e:
        print("  (sin red:", str(e)[:60], "-> guitarra sintética)")
    # 3) respaldo sintético: arpegio
    t = np.arange(int(seconds * SR)) / SR
    notes = [196, 247, 294, 392, 294, 247]
    x = np.zeros_like(t)
    seg = len(t) // len(notes)
    for i, f in enumerate(notes):
        s = i * seg
        x[s:s + seg] += np.sin(2 * np.pi * f * t[s:s + seg]) * np.hanning(seg)
    return normalize_audio(x), "sintética"

guitar, src = load_guitar()
print("Guitarra (%s): %.2f s" % (src, len(guitar) / SR))
plt.figure(figsize=(12, 2.5)); plt.plot(np.arange(len(guitar)) / SR, guitar, lw=0.5)
plt.title("Guitarra de entrada (%s)" % src); plt.xlabel("s"); plt.grid(alpha=0.3); plt.show()
display(Audio(guitar, rate=SR))
"""))

cells.append(md("## 4. La voz\n"
                "El cuaderno trae **incluida una voz real de Vocaloid 4** (CyberDiva, voz femenina "
                "cantada — uso académico, declarado), así que suena a voz de verdad sin subir nada. "
                "Si quieres, puedes subir tu propia muestra (`.wav`, un \"aaah\" sostenido); y si todo "
                "falla, hay una vocal sintética de respaldo."))
cells.append(code('VOICE_B64 = "%s"' % VOICE_B64))
cells.append(code(r"""import base64, io
def load_voice():
    # 1) subir en Colab (override manual)
    if IN_COLAB:
        try:
            from google.colab import files
            print("Sube tu voz (.wav) o pulsa Cancelar para usar la voz incluida (CyberDiva V4).")
            up = files.upload()
            if up:
                import librosa
                y, _ = librosa.load(list(up.keys())[0], sr=SR, mono=True)
                return ensure_mono_float(y), "muestra subida"
        except Exception as e:
            print("  (sin upload:", str(e)[:60], ")")
    # 2) voz real incluida (embebida) o copia local
    if VOICE_B64:
        try:
            import soundfile as sf
            y, _ = sf.read(io.BytesIO(base64.b64decode(VOICE_B64)))
            return ensure_mono_float(y), "voz incluida (CyberDiva V4)"
        except Exception as e:
            print("  (voz embebida no disponible:", str(e)[:60], ")")
    for p in ("voces/miku_voice.wav", "miku_voice.wav"):
        if os.path.exists(p):
            import librosa
            y, _ = librosa.load(p, sr=SR, mono=True)
            return ensure_mono_float(y), "local: " + p
    # 3) respaldo sintético
    return synth_vowel(200.0, "a", dur=0.6, seed=2026), "vocal sintética /a/"

voice_raw, vsrc = load_voice()
voice_grain, voice_f0 = prep_voice_grain(voice_raw, SR)
print("Voz (%s): grano %.0f ms, f0 base = %.0f Hz" % (vsrc, 1000 * len(voice_grain) / SR, voice_f0))
f, mag = compute_rfft(voice_grain)
plt.figure(figsize=(12, 2.6)); plt.plot(f, mag / (mag.max() + 1e-9), lw=0.8, color="tab:purple")
plt.xlim(0, 3500); plt.title("Espectro de la voz (formantes)"); plt.xlabel("Hz"); plt.grid(alpha=0.3); plt.show()
display(Audio(voice_grain, rate=SR))
"""))

cells.append(md("## 5. Detección de notas (pYIN, monofónico)\n"
                "Estimamos el contorno de f0 y los tramos sonoros, y los agrupamos en notas."))
cells.append(code(r"""times, f0, voiced = track_f0_pyin(guitar, SR)
notes = segment_notes(times, f0, voiced)
print("Notas detectadas:", len(notes))

# espectrograma + contorno de f0
X = stft(guitar, n_fft=1024, hop=256)
S = 20 * np.log10(np.abs(X).T + 1e-6)
ff = np.fft.rfftfreq(1024, 1.0 / SR); tt = np.arange(X.shape[0]) * 256 / SR
m = ff <= 2000
plt.figure(figsize=(12, 4))
plt.pcolormesh(tt, ff[m], S[m], shading="auto", cmap="magma", vmin=S.max() - 70, vmax=S.max())
plt.plot(times, f0, ".", ms=4, color="cyan", label="f0 (pYIN)")
for (a, b, c) in notes:
    plt.hlines(c, a, b, color="white", lw=2)
plt.ylim(0, 2000); plt.title("Contorno de f0 y notas sobre el espectrograma")
plt.xlabel("tiempo [s]"); plt.ylabel("Hz"); plt.legend(loc="upper right"); plt.show()
"""))

cells.append(md("## 6. Síntesis: 3 modos (robótico vs limpio)\n"
                "La voz **sigue el contorno de tono** de la guitarra (glissando continuo), transpuesto "
                "a un registro cantable. Modos:\n"
                "- **resample** — afinado por remuestreo (corre formantes): suena robótico, **como el "
                "pedal**.\n- **pv** — phase vocoder.\n- **pv_formant** — phase vocoder con formantes: "
                "mantiene el timbre, **mejor que el pedal**.\nEscucha el A/B."))
cells.append(code(r"""voices = {}
for method in ["resample", "pv", "pv_formant"]:
    voices[method] = miku_stomp_glide(times, f0, voiced, SR, voice_grain, voice_f0,
                                      len(guitar), method=method)

labels = {"resample": "Remuestreo (robótico, como el pedal)",
          "pv": "Phase vocoder",
          "pv_formant": "PV + formantes (limpio, mejor que el pedal)"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
for a, mth in zip(ax, voices):
    Xv = stft(voices[mth], n_fft=1024, hop=256)
    Sv = 20 * np.log10(np.abs(Xv).T + 1e-6)
    a.pcolormesh(np.arange(Xv.shape[0]) * 256 / SR, ff[m], Sv[m], shading="auto",
                 cmap="magma", vmin=Sv.max() - 70, vmax=Sv.max())
    a.set_title(labels[mth], fontsize=9); a.set_xlabel("s"); a.set_ylabel("Hz")
plt.tight_layout(); plt.show()
for mth in voices:
    print(labels[mth]); display(Audio(normalize_audio(voices[mth]), rate=SR))
"""))

cells.append(md("## 7. Perilla MIX y salida final\n"
                "Mezcla la voz con la guitarra seca (como el pedal) y descarga el resultado."))
cells.append(code(r"""MIX = 0.6   # 0 = solo guitarra, 1 = solo voz
final = mix_over(guitar, voices["pv_formant"], mix=MIX)

import soundfile as sf
out_name = "miku_stomp_salida.wav"
sf.write(out_name, normalize_audio(final), SR)
print("Guardado:", out_name)
plt.figure(figsize=(12, 2.6)); plt.plot(np.arange(len(final)) / SR, final, lw=0.5, color="tab:green")
plt.title("Salida final (guitarra + voz Miku, MIX=%.1f)" % MIX); plt.xlabel("s"); plt.grid(alpha=0.3); plt.show()
display(Audio(final, rate=SR))
if IN_COLAB:
    try:
        from google.colab import files
        files.download(out_name)
    except Exception as e:
        print("(descarga manual desde el panel de archivos)", str(e)[:60])
"""))

cells.append(md(r"""## 8. Discusión

- **Por qué supera al pedal:** el Korg Miku Stomp es monofónico y al transponer **corre los
  formantes**; nuestro modo `pv_formant` **mantiene el timbre** (formantes fijos) y afina con pYIN,
  más preciso que el seguimiento analógico del pedal.
- **Límites:** es **monofónico** (una nota a la vez) y funciona por **lotes** (graba → procesa →
  descarga), no en tiempo real como el pedal físico. En notas muy agudas el plegado de octava mantiene
  la voz cantable, pero saltos extremos degradan el phase vocoder (igual que en el proyecto del
  artículo).
- **Trabajo futuro (fuera del eje DSP del ramo):** polifonía con *Basic Pitch*, clonación de timbre
  con *RVC*, o síntesis tipo *OpenUTAU*; y llevarlo a **tiempo real** como plugin (JUCE / Reaper).
- **Reutilización:** el núcleo DSP (`vocoder.py`) proviene del proyecto del phase vocoder; lo nuevo
  aquí es el pipeline del pedal (`stomp.py`: pYIN, segmentación de notas y síntesis con tiling).
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "authors": [{"name": "Francisco Alejandro Pinto Abraham"}],
}
out = os.path.join(HERE, "miku_stomp_colab.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print("Notebook escrito:", out, "(%d celdas)" % len(cells))
