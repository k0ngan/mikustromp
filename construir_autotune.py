# -*- coding: utf-8 -*-
"""
construir_autotune.py -- Arma `miku_autotune_colab.ipynb`, un cuaderno AUTOCONTENIDO (Colab o local)
que implementa el "Autotune con voz de Miku" en DSP puro.

Embebe el núcleo DSP de `vocoder.py`, el detector de tono de `stomp.py` y el motor de `autotune.py`
en celdas (en Colab no hace falta subir esos archivos). Incluye la voz real de Miku como base64.
Funciona **en vivo con audios nuevos**: puedes **subir** tu voz, **grabar con el micrófono** en Colab,
o usar una voz de prueba. Las celdas específicas de Colab van guardadas por `IN_COLAB`, así el cuaderno
también corre de punta a punta en local con respaldos.
"""

import os
import base64
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))

# Voz real de Miku (Vocaloid 4 CyberDiva) embebida como base64 -> es la REFERENCIA de timbre.
_voice_path = os.path.join(HERE, 'voces', 'miku_voice.wav')
VOICE_B64 = base64.b64encode(open(_voice_path, 'rb').read()).decode('ascii') if os.path.exists(_voice_path) else ''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


# --- núcleo DSP embebido: vocoder + stomp (detector) + autotune (motor) -----------------
vocoder_src = open(os.path.join(HERE, 'vocoder.py'), encoding='utf-8').read()

stomp_src = open(os.path.join(HERE, 'stomp.py'), encoding='utf-8').read()
stomp_src = stomp_src.replace('import vocoder as V\n', '')
stomp_src = stomp_src.replace('V.', '')

autotune_src = open(os.path.join(HERE, 'autotune.py'), encoding='utf-8').read()
autotune_src = autotune_src.replace('import vocoder as V\n', '')
autotune_src = autotune_src.replace('from stomp import track_f0_pyin           # detección de tono (pYIN clásico, con respaldo autocorr)\n', '')
autotune_src = autotune_src.replace('from stomp import track_f0_pyin', '')
autotune_src = autotune_src.replace('V.', '')

cells = []

cells.append(md(r"""# Autotune con voz de Miku — corregir la afinación y ponerle timbre de Miku (DSP puro)

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1**
Francisco Alejandro Pinto Abraham · RUT 21.571.239-7

Este cuaderno es un **autotune**: toma **tu voz** cantada, (1) sigue su **tono** cuadro a cuadro,
(2) lo **pega a las notas** de una escala musical (el "snap" del autotune), (3) corrige el tono de
forma **variable en el tiempo** con un **phase vocoder** que **preserva los formantes**, y
(4) le transfiere el **timbre (formantes) de Hatsune Miku**. Resultado: tu misma interpretación, pero
**afinada** y con **voz de Miku**.

**Regla del ramo:** todo es DSP clásico — **FFT** (rfft/irfft), **STFT**/enventanado Hann,
**phase vocoder** (Dolson 1986), **cepstrum** (formantes), **overlap-add**, **autocorrelación/pYIN**
para el tono. **Sin** deep learning ni IA generativa (RVC, so-vits, etc. quedan solo como *trabajo futuro*).

> **En vivo con audios nuevos:** en Colab puedes **subir** un `.wav`/`.mp3` de tu voz o **grabar con el
> micrófono** (`USE_MIC = True`). En local usa `LOCAL_PATH` o corre con la voz de prueba. Trae **incluida
> la voz real de Miku** (Vocaloid 4 CyberDiva) como referencia de timbre, así suena a Miku sin subir nada.
> **Colab: Entorno de ejecución → Ejecutar todo.**
"""))

cells.append(md("## 0. Dependencias"))
cells.append(code(r"""try:
    import google.colab  # noqa
    !pip install -q librosa soundfile numpy scipy matplotlib
except Exception:
    pass
print("Dependencias listas.")
"""))

cells.append(code(r"""import os, io, base64, urllib.request, warnings
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
                "Mismas funciones validadas en `vocoder.py` (Dolson 1986 + preservación de formantes)."))
cells.append(code(vocoder_src))

cells.append(md("## 2. Detector de tono (pYIN / autocorrelación)\n"
                "De `stomp.py`: `track_f0_pyin` sigue el contorno de f0 (pYIN clásico, con respaldo por "
                "autocorrelación si no hay librosa)."))
cells.append(code(stomp_src))

cells.append(md("## 3. Motor del Autotune (snap a la escala + corrección + timbre Miku)\n"
                "De `autotune.py`: `snap_midi_to_scale`, `correction_semitones`, `autotune_voice`, "
                "`miku_formant_template`, `miku_formant_transfer` y el pipeline `mikutune`."))
cells.append(code(autotune_src))

cells.append(md("## 4. La voz de referencia de Miku (timbre objetivo)\n"
                "Incluida y embebida (Vocaloid 4 CyberDiva, uso académico declarado). De aquí sale la "
                "**envolvente de formantes** que le da a tu voz el color de Miku."))
cells.append(code('VOICE_B64 = "%s"' % VOICE_B64))
cells.append(code(r"""def load_miku_ref():
    if VOICE_B64:
        try:
            import soundfile as sf
            y, _ = sf.read(io.BytesIO(base64.b64decode(VOICE_B64)))
            return normalize_audio(ensure_mono_float(y)), "voz Miku incluida (CyberDiva V4)"
        except Exception as e:
            print("  (voz embebida no disponible:", str(e)[:60], ")")
    for p in ("voces/miku_voice.wav", "miku_voice.wav"):
        if os.path.exists(p):
            import librosa
            y, _ = librosa.load(p, sr=SR, mono=True)
            return normalize_audio(ensure_mono_float(y)), "local: " + p
    return synth_vowel(520.0, "i", dur=0.6, seed=2026), "vocal sintética (respaldo)"

miku_ref, mref = load_miku_ref()
print("Referencia de timbre:", mref, "(%.2f s)" % (len(miku_ref) / SR))
f, mag = compute_rfft(miku_ref)
plt.figure(figsize=(12, 2.6)); plt.plot(f, mag / (mag.max() + 1e-9), lw=0.8, color="tab:purple")
plt.xlim(0, 4000); plt.title("Espectro de la voz de Miku (formantes objetivo)"); plt.xlabel("Hz")
plt.grid(alpha=0.3); plt.show()
display(Audio(miku_ref, rate=SR))
"""))

cells.append(md("## 5. Tu voz de entrada — **en vivo con audios nuevos**\n"
                "Elige la fuente:\n"
                "- **Subir** un archivo (Colab): deja `USE_MIC = False` y sube tu `.wav`/`.mp3`.\n"
                "- **Micrófono en vivo** (Colab): pon `USE_MIC = True` y canta al ejecutar la celda.\n"
                "- **Local:** pon la ruta en `LOCAL_PATH`; si queda vacío, se usa una **voz de prueba** "
                "desafinada para ver el efecto."))
cells.append(code(r"""# ------------------ configuración de la entrada ------------------
USE_MIC        = False     # Colab: True para grabar tu voz en vivo con el micrófono
RECORD_SECONDS = 6.0       # duración de la grabación (si USE_MIC)
LOCAL_PATH     = ""        # local: ruta a tu voz .wav/.mp3 (vacío = voz de prueba)


def _record_colab(seconds, sr):
    '''Graba unos segundos del microfono en Colab (getUserMedia). Es E/S (captura del navegador),
    no materia del curso; el audio se decodifica y se remuestrea por interpolacion.'''
    from google.colab import output as colab_output
    from base64 import b64decode
    js = '''
    async function recordMic(sec){
      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      const rec = new MediaRecorder(stream); const chunks = [];
      rec.ondataavailable = e => chunks.push(e.data);
      const stopped = new Promise(r => rec.onstop = r);
      rec.start(); await new Promise(r => setTimeout(r, sec*1000));
      rec.stop(); await stopped; stream.getTracks().forEach(t => t.stop());
      const buf = await (new Blob(chunks)).arrayBuffer();
      const bytes = new Uint8Array(buf); let bin='';
      for (let i=0;i<bytes.length;i++) bin += String.fromCharCode(bytes[i]);
      return btoa(bin);
    }'''
    colab_output.eval_js(js)
    print("Grabando %.1f s... ¡canta ahora!" % seconds)
    b64 = colab_output.eval_js('recordMic(%f)' % float(seconds))
    open("mic_input.webm", "wb").write(b64decode(b64))
    import librosa
    y, _ = librosa.load("mic_input.webm", sr=sr, mono=True)
    return ensure_mono_float(y)


def synth_offkey_demo(seed=2026):
    '''Voz de prueba desafinada (Do mayor con +-30..50 cents de error) para demostrar el autotune.'''
    rng = np.random.default_rng(seed)
    midis = [60, 62, 64, 65, 67, 69, 67, 64]; vows = ["a","e","i","o","a","e","i","o"]
    parts = []
    for m, vw in zip(midis, vows):
        dt = float(rng.uniform(0.30, 0.5)) * (1 if rng.random() > 0.5 else -1)
        f = midi_to_freq(m + dt)
        parts.append(synth_vowel(f0=float(f), vowel=vw, dur=0.6, fs=SR, seed=int(rng.integers(1, 10**6))))
    return normalize_audio(np.concatenate(parts))


def load_input():
    if IN_COLAB and USE_MIC:
        try:
            return _record_colab(RECORD_SECONDS, SR), "micrófono en vivo"
        except Exception as e:
            print("  (sin micrófono:", str(e)[:60], ")")
    if IN_COLAB and not USE_MIC:
        try:
            from google.colab import files
            print("Sube tu voz (.wav/.mp3); o pulsa Cancelar para usar la voz de prueba.")
            up = files.upload()
            if up:
                import librosa
                y, _ = librosa.load(list(up.keys())[0], sr=SR, mono=True)
                return ensure_mono_float(y), "archivo subido"
        except Exception as e:
            print("  (sin upload:", str(e)[:60], ")")
    if LOCAL_PATH and os.path.exists(LOCAL_PATH):
        return load_audio(LOCAL_PATH, SR), "local: " + LOCAL_PATH
    return synth_offkey_demo(), "voz de prueba (desafinada)"


voz, vsrc = load_input()
print("Entrada (%s): %.2f s" % (vsrc, len(voz) / SR))
plt.figure(figsize=(12, 2.4)); plt.plot(np.arange(len(voz)) / SR, voz, lw=0.5)
plt.title("Tu voz de entrada (%s)" % vsrc); plt.xlabel("s"); plt.grid(alpha=0.3); plt.show()
display(Audio(voz, rate=SR))
"""))

cells.append(md("## 6. Perillas del autotune\n"
                "- **KEY / SCALE:** tonalidad y escala a la que se pega el tono (`chromatic`, `major`, "
                "`minor`, `pentatonic`).\n"
                "- **STRENGTH** (0–1): 0 no corrige, 1 pega del todo (efecto duro).\n"
                "- **RETUNE_SPEED** (0–1): 1 = enganche instantáneo (robótico); bajo = glissando suave.\n"
                "- **MIKU_AMOUNT** (0–1): cuánto timbre de Miku se impone.\n"
                "- **OCTAVE:** transpone por octavas (p. ej. +1 para el registro agudo de Miku)."))
cells.append(code(r"""KEY          = "C"          # tónica: C, D, E, F, G, A, B (con # o b)
SCALE        = "major"      # chromatic | major | minor | pentatonic
STRENGTH     = 1.0          # 0..1
RETUNE_SPEED = 1.0          # 0..1  (1 = instantáneo)
MIKU_AMOUNT  = 0.9          # 0..1
OCTAVE       = 0            # octavas de transposición

res = mikutune(voz, SR, key=KEY, scale=SCALE, strength=STRENGTH, retune_speed=RETUNE_SPEED,
               octave=OCTAVE, miku_amount=MIKU_AMOUNT, method="pv_formant", ref_miku=miku_ref)
auto, miku, info = res["auto"], res["miku"], res["info"]
print("Autotune listo. Notas de la escala %s de %s aplicadas." % (SCALE, KEY))
"""))

cells.append(md("## 7. Qué pasó con el tono (snap a la escala)\n"
                "El tono medido (gris) se pega a las notas de la escala (rojo); el tono corregido (azul) "
                "queda sobre ellas."))
cells.append(code(r"""t = info["times"]
plt.figure(figsize=(12, 4))
plt.plot(t, info["orig_midi"], ".", ms=4, color="tab:gray", label="tono medido (tu voz)")
plt.plot(t, info["snapped_midi"], "_", ms=9, color="tab:red", label="nota de la escala (snap)")
corr = np.where(info["voiced"], info["orig_midi"] + info["corr"], np.nan)
plt.plot(t, corr, ".", ms=3, color="tab:blue", label="tono corregido (autotune)")
kpc = note_to_pitch_class(KEY); sset = set(SCALES[SCALE])
lo = int(np.nanmin(info["orig_midi"])) - 3; hi = int(np.nanmax(info["orig_midi"])) + 3
for m in range(lo, hi):
    if ((m - kpc) % 12) in sset:
        plt.axhline(m, color="0.9", lw=0.6, zorder=0)
plt.xlabel("tiempo [s]"); plt.ylabel("nota (MIDI)")
plt.title("Autotune: el tono se pega a la escala de %s %s" % (KEY, SCALE))
plt.legend(loc="upper right"); plt.tight_layout(); plt.show()
"""))

cells.append(md("## 8. Escucha el A/B y descarga\n"
                "**Entrada** (tu voz) · **Autotune** (afinada, tu timbre) · **Autotune + Miku** (afinada "
                "con voz de Miku). Compara los espectrogramas."))
cells.append(code(r"""fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
for a, sig, ttl in [(ax[0], voz, "Entrada"), (ax[1], auto, "Autotune"),
                    (ax[2], miku, "Autotune + Miku")]:
    X = stft(sig, n_fft=1024, hop=256); S = 20*np.log10(np.abs(X).T + 1e-6)
    ff = np.fft.rfftfreq(1024, 1.0/SR); mm = ff <= 4000
    a.pcolormesh(np.arange(X.shape[0])*256/SR, ff[mm], S[mm], shading="auto",
                 cmap="magma", vmin=S.max()-70, vmax=S.max())
    a.set_title(ttl); a.set_xlabel("s"); a.set_ylabel("Hz")
plt.tight_layout(); plt.show()

print("Entrada (tu voz):");            display(Audio(normalize_audio(voz), rate=SR))
print("Autotune (afinada, tu timbre):"); display(Audio(normalize_audio(auto), rate=SR))
print("Autotune + voz de Miku:");       display(Audio(normalize_audio(miku), rate=SR))

import soundfile as sf
out_name = "miku_autotune_salida.wav"
sf.write(out_name, normalize_audio(miku), SR)
print("Guardado:", out_name)
if IN_COLAB:
    try:
        from google.colab import files
        files.download(out_name)
    except Exception as e:
        print("(descarga manual desde el panel de archivos)", str(e)[:60])
"""))

cells.append(md(r"""## 9. Discusión

- **Cómo funciona el "snap":** se estima el tono f0 (pYIN/autocorrelación), se pasa a número MIDI
  (`m = 69 + 12·log2(f/440)`) y se redondea a la **nota más cercana de la escala**; la diferencia en
  semitonos se corrige con el **phase vocoder preservando formantes** (así la vocal no se vuelve
  "ardilla" al subir el tono). Como se corrige por **bloques con overlap-add**, la **duración se
  conserva** (no se acelera ni frena la voz).
- **Por qué suena a Miku:** se estima la **envolvente de formantes** de una voz real de Miku (por
  **cepstrum**) y se **impone** sobre tu voz afinada como una máscara espectral `H = env_Miku/env_tuya`.
  El tono es tuyo (tu melodía), pero el **color** (formantes) es de Miku.
- **Efecto T-Pain:** con `RETUNE_SPEED = 1` y `STRENGTH = 1` el enganche es instantáneo (robótico);
  bajando `RETUNE_SPEED` el tono llega a la nota con un **glissando** más natural.
- **Límites:** es **monofónico** (una voz a la vez) y funciona **por lotes** (graba/sube → procesa →
  reproduce), no en streaming de latencia cero como un plugin en vivo. Saltos de tono muy grandes
  degradan el phase vocoder (por eso se acota el desplazamiento).
- **Trabajo futuro (fuera del eje DSP del ramo):** clonación de timbre con RVC/so-vits, síntesis tipo
  OpenUTAU, y una versión en **tiempo real** por bloques (`sounddevice`) o como plugin (JUCE/Reaper).
- **Reutilización:** el núcleo DSP (`vocoder.py`) viene del proyecto del phase vocoder (artículo de
  Dolson 1986); lo nuevo aquí es el **motor de autotune** (`autotune.py`: snap a escala, corrección
  variable en el tiempo y transferencia de formantes de Miku).
"""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "authors": [{"name": "Francisco Alejandro Pinto Abraham"}],
}
out = os.path.join(HERE, "miku_autotune_colab.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    nbf.write(nb, fh)
print("Notebook escrito:", out, "(%d celdas)" % len(cells))
