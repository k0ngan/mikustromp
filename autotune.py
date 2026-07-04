# -*- coding: utf-8 -*-
"""
autotune.py -- "Autotune con voz de Miku": corrección de afinación en tiempo real (por lotes) sobre
TU voz, más transferencia del timbre (formantes) de Hatsune Miku. DSP puro, sin ML / deep learning.

Procesamiento Digital de Señales e Imágenes (INFB6063) -- UTEM 2026-1
Estudiante: Francisco Alejandro Pinto Abraham -- RUT 21.571.239-7

Idea: tomar una voz cantada, (1) estimar su contorno de tono f0 cuadro a cuadro, (2) "pegar" cada
tono a la nota más cercana de una ESCALA musical (el "snap" del autotune), (3) corregir el tono de
forma VARIABLE EN EL TIEMPO con el phase vocoder preservando formantes, y (4) reemplazar la envolvente
de formantes por la de Miku, de modo que la salida suena AFINADA y con el TIMBRE de Miku, pero conserva
tu interpretación (tu melodía y tu ritmo).

Todo reutiliza el núcleo DSP validado en `vocoder.py`:
  - `track_f0_pyin` / `estimate_f0_autocorr`  -> seguimiento de tono (pYIN clásico o autocorrelación),
  - `pitch_shift_pv_formant` (phase vocoder + formantes)  -> corrección de tono sin mover el timbre,
  - `cepstral_envelope`  -> plantilla de formantes de Miku,
  - `stft`/`istft`, ventana Hann y overlap-add  -> procesamiento por bloques.

Materia del curso: DFT/FFT (rfft/irfft), STFT/enventanado, overlap-add, cepstrum, autocorrelación,
muestreo/interpolación. Sin redes neuronales ni IA generativa.
"""

import numpy as np
import vocoder as V
from stomp import track_f0_pyin           # detección de tono (pYIN clásico, con respaldo autocorr)

SR = V.SR
EPS = 1e-9
A4 = 440.0                                 # referencia de afinación (La4 = 440 Hz)

# Grados (semitonos desde la tónica) que pertenecen a cada escala.
SCALES = {
    'chromatic':  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'major':      [0, 2, 4, 5, 7, 9, 11],       # mayor (jónica)
    'minor':      [0, 2, 3, 5, 7, 8, 10],       # menor natural (eólica)
    'pentatonic': [0, 2, 4, 7, 9],              # pentatónica mayor (típica del "efecto Miku")
}

# Nombre de nota -> clase de altura (0..11). Acepta sostenidos y bemoles.
_PITCH_CLASS = {
    'C': 0, 'C#': 1, 'DB': 1, 'D': 2, 'D#': 3, 'EB': 3, 'E': 4, 'F': 5, 'F#': 6, 'GB': 6,
    'G': 7, 'G#': 8, 'AB': 8, 'A': 9, 'A#': 10, 'BB': 10, 'B': 11,
}


# =====================================================================================
# 1) CONVERSIONES TONO <-> NOTA  (materia: muestreo/logaritmos de frecuencia)
# =====================================================================================
def freq_to_midi(f):
    """Frecuencia (Hz) -> número MIDI continuo: m = 69 + 12·log2(f/440)."""
    f = np.asarray(f, dtype=float)
    return 69.0 + 12.0 * np.log2(np.maximum(f, EPS) / A4)


def midi_to_freq(m):
    """Número MIDI -> frecuencia (Hz): f = 440·2^((m-69)/12)."""
    return A4 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def note_to_pitch_class(name):
    """'C', 'A#', 'Bb', 'F#4' -> clase de altura 0..11 (ignora la octava)."""
    n = str(name).strip().upper().replace('♯', '#').replace('♭', 'B')
    n = n.rstrip('0123456789')
    return _PITCH_CLASS.get(n, 0)


def snap_midi_to_scale(midi, key_pc, scale_set):
    """Qué hace: redondea un tono MIDI continuo a la NOTA de la escala más cercana (el "snap").
    Matemática: entre los MIDI enteros m con ((m - tónica) mod 12) en la escala, elige el de menor |m-midi|.
    """
    if not np.isfinite(midi):
        return np.nan
    base = int(np.round(midi))
    best, best_d = None, 1e9
    for m in range(base - 7, base + 8):
        if ((m - key_pc) % 12) in scale_set:
            d = abs(m - midi)
            if d < best_d:
                best_d, best = d, m
    return float(best if best is not None else base)


# =====================================================================================
# 2) CONTORNO DE CORRECCIÓN  (el "motor" del autotune: cuánto corregir en cada instante)
# =====================================================================================
def correction_semitones(f0, voiced, key_pc, scale_set,
                         strength=1.0, retune_speed=1.0, octave=0):
    """Qué hace: a partir del contorno de f0, calcula el DESPLAZAMIENTO en semitonos que hay que aplicar
    cuadro a cuadro para llevar cada tono a su nota de la escala.
    Curso: logaritmos de frecuencia + suavizado (filtro de un polo).
    Parámetros de autotune:
      - strength   [0..1]: 0 = no corrige (bypass), 1 = pega totalmente a la nota (efecto duro T-Pain).
      - retune_speed [0..1]: velocidad de enganche. 1 = instantáneo (robótico); pequeño = glissando suave.
      - octave (entero): transpone todo por octavas (p. ej. +1 para el registro agudo de Miku).
    Devuelve (corr, orig_midi, snapped_midi, target_midi).
    """
    f0 = np.asarray(f0, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    orig = freq_to_midi(f0)                         # tono medido (MIDI continuo)
    snapped = np.full_like(orig, np.nan)
    for i in range(orig.size):
        if voiced[i] and np.isfinite(orig[i]):
            snapped[i] = snap_midi_to_scale(orig[i], key_pc, scale_set)
    # objetivo = mezcla entre el tono medido y la nota pegada, más la transposición por octavas
    target = orig + strength * (snapped - orig) + 12.0 * int(octave)

    # suavizado temporal (filtro de un polo) para modelar la "velocidad de enganche"; se reinicia
    # en cada tramo sin voz para no arrastrar tono a través de los silencios.
    alpha = float(np.clip(retune_speed, 0.02, 1.0))
    smoothed = np.copy(target)
    prev = np.nan
    for i in range(target.size):
        if voiced[i] and np.isfinite(target[i]):
            prev = target[i] if not np.isfinite(prev) else prev + alpha * (target[i] - prev)
            smoothed[i] = prev
        else:
            prev = np.nan
    corr = smoothed - orig                          # semitonos a corregir en cada cuadro
    corr = np.where(voiced & np.isfinite(corr), corr, 0.0)
    return corr, orig, snapped, target


# =====================================================================================
# 3) CORRECCIÓN DE TONO VARIABLE EN EL TIEMPO  (phase vocoder por bloques + overlap-add)
# =====================================================================================
def autotune_voice(x, sr=SR, key='C', scale='major', strength=1.0, retune_speed=1.0,
                   octave=0, method='pv_formant', frame=2048, hop=512,
                   fmin=80.0, fmax=1000.0, max_shift=12.0):
    """Qué hace: AFINA una voz a una escala, corrigiendo el tono de forma variable en el tiempo.
    Curso: STFT/enventanado (Hann), phase vocoder (Dolson), overlap-add.
    Método: se enmarca la señal con ventana Hann y 75% de solape; para cada cuadro se estima cuánto
    corregir (`correction_semitones`) y se aplica `pitch_shift_pv_formant` (mantiene los formantes de TU
    voz mientras cambia el tono); la salida se reconstruye por superposición-suma normalizada por la
    ventana al cuadrado (WOLA). La DURACIÓN se conserva (no se estira el tiempo).
    Devuelve (audio_afinado, info) con info = contornos para graficar.
    """
    x = V.ensure_mono_float(x)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    win = np.hanning(frame).astype(np.float64)

    # contorno de tono y de corrección
    times, f0, voiced = track_f0_pyin(x, sr, fmin=fmin, fmax=fmax, frame=frame, hop=hop)
    key_pc = note_to_pitch_class(key)
    scale_set = set(SCALES.get(scale, SCALES['chromatic']))
    corr, orig_midi, snapped, target = correction_semitones(
        f0, voiced, key_pc, scale_set, strength, retune_speed, octave)

    # procesamiento por bloques con overlap-add
    out = np.zeros(x.size + frame, dtype=np.float64)
    wsum = np.zeros_like(out)
    n_frames = 1 + (x.size - frame) // hop
    tv = np.asarray(times)
    applied = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        tc = (s + frame / 2.0) / sr                 # centro del cuadro en segundos
        j = int(np.argmin(np.abs(tv - tc))) if tv.size else 0
        n_semi = float(corr[j]) if j < corr.size else 0.0
        n_semi = float(np.clip(n_semi, -max_shift, max_shift))
        applied[i] = n_semi
        seg = x[s:s + frame] * win
        if abs(n_semi) < 0.03:                       # sin corrección apreciable -> ahorra cómputo
            shifted = seg
        else:
            sh = V.PITCH_METHODS[method](seg.astype(np.float32), n_semi)
            shifted = np.zeros(frame, dtype=np.float64)
            L = min(frame, sh.size)
            shifted[:L] = sh[:L]
        out[s:s + frame] += shifted * win
        wsum[s:s + frame] += win ** 2
    y = out[:x.size] / np.where(wsum[:x.size] > EPS, wsum[:x.size], 1.0)

    info = {'times': times, 'f0': f0, 'voiced': voiced, 'orig_midi': orig_midi,
            'snapped_midi': snapped, 'target_midi': target, 'corr': corr, 'applied': applied}
    return V.normalize_audio(y), info


# =====================================================================================
# 4) TRANSFERENCIA DE TIMBRE DE MIKU  (envolvente cepstral -> "imitar a Miku por frecuencia")
# =====================================================================================
def miku_formant_template(ref, sr=SR, n_fft=2048, hop=1024, n_lifter=30):
    """Qué hace: estima la ENVOLVENTE de formantes PROMEDIO de una muestra de voz de Miku, para usarla
    como plantilla de timbre. Curso: STFT + envolvente cepstral (cepstrum).
    Promedia en el dominio logarítmico los cuadros con energía suficiente y normaliza la ganancia
    (queda solo la FORMA de los formantes). Devuelve (freqs_Hz, plantilla) sobre la grilla de `n_fft`.
    """
    ref = V.normalize_audio(V.ensure_mono_float(ref))
    if ref.size < n_fft:
        ref = np.pad(ref, (0, n_fft - ref.size))
    win = np.hanning(n_fft).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    log_envs, energies = [], []
    for s in range(0, ref.size - n_fft + 1, hop):
        fr = ref[s:s + n_fft] * win
        energies.append(float(np.sqrt(np.mean(fr ** 2))))
        mag = np.abs(np.fft.rfft(fr))
        log_envs.append(np.log(V.cepstral_envelope(mag, n_lifter) + EPS))
    if not log_envs:
        mag = np.abs(np.fft.rfft(ref[:n_fft] * win))
        return freqs, V.cepstral_envelope(mag, n_lifter)
    energies = np.array(energies)
    thr = 0.3 * float(energies.max() + EPS)
    sel = [le for le, e in zip(log_envs, energies) if e >= thr] or log_envs
    tmpl = np.exp(np.mean(np.array(sel), axis=0))
    tmpl = tmpl / (np.mean(tmpl) + EPS)             # solo forma (independiente de la ganancia)
    return freqs, tmpl.astype(np.float64)


def miku_formant_transfer(x, sr=SR, ref=None, template=None, amount=1.0,
                          frame=2048, hop=512, n_lifter=30):
    """Qué hace: reemplaza la envolvente de formantes de `x` por la de Miku (plantilla), cuadro a cuadro.
    Curso: STFT, filtrado como máscara espectral H[k], cepstrum, overlap-add.
    Matemática: por cuadro, env_x = envolvente cepstral de |X|; H = (plantilla/env_x)^amount, acotada a
    ±40 dB; y = IDFT(X·H). amount=0 no cambia nada; amount=1 impone del todo el timbre de Miku.
    """
    x = V.ensure_mono_float(x)
    if template is None:
        template = miku_formant_template(ref, sr, n_fft=frame, hop=frame // 2, n_lifter=n_lifter)
    freqs_t, tmpl = template
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    win = np.hanning(frame).astype(np.float64)
    tf = np.fft.rfftfreq(frame, 1.0 / sr)
    tmpl_i = np.interp(tf, freqs_t, tmpl)           # plantilla en la grilla del cuadro
    tmpl_i = tmpl_i / (np.mean(tmpl_i) + EPS)
    amount = float(np.clip(amount, 0.0, 1.0))
    out = np.zeros(x.size + frame, dtype=np.float64)
    wsum = np.zeros_like(out)
    n_frames = 1 + (x.size - frame) // hop
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s + frame] * win
        spec = np.fft.rfft(seg)
        env = V.cepstral_envelope(np.abs(spec), n_lifter)
        env = env / (np.mean(env) + EPS)
        H = np.clip(tmpl_i / (env + EPS), 1e-2, 1e2) ** amount
        y = np.fft.irfft(spec * H, n=frame)
        out[s:s + frame] += y * win
        wsum[s:s + frame] += win ** 2
    y = out[:x.size] / np.where(wsum[:x.size] > EPS, wsum[:x.size], 1.0)
    return V.normalize_audio(y)


# =====================================================================================
# 5) PIPELINE COMPLETO  (autotune + timbre Miku)
# =====================================================================================
def mikutune(x, sr=SR, key='C', scale='major', strength=1.0, retune_speed=1.0, octave=0,
             miku_amount=0.85, method='pv_formant', ref_miku=None, frame=2048, hop=512,
             fmin=80.0, fmax=1000.0):
    """Qué hace: pipeline completo del "Autotune Miku". Afina la voz a la escala y luego le transfiere
    el timbre de Miku. Devuelve un dict con la entrada, la etapa afinada, la salida con voz Miku, la
    plantilla de formantes, los contornos (para graficar) y los parámetros usados.
    """
    x = V.ensure_mono_float(x)
    auto, info = autotune_voice(x, sr, key=key, scale=scale, strength=strength,
                                retune_speed=retune_speed, octave=octave, method=method,
                                frame=frame, hop=hop, fmin=fmin, fmax=fmax)
    template = None
    miku = auto
    if ref_miku is not None and miku_amount > 0:
        template = miku_formant_template(ref_miku, sr, n_fft=frame, hop=frame // 2)
        miku = miku_formant_transfer(auto, sr, template=template, amount=miku_amount,
                                     frame=frame, hop=hop)
    return {'input': x, 'auto': auto, 'miku': miku, 'template': template, 'info': info,
            'params': {'key': key, 'scale': scale, 'strength': strength,
                       'retune_speed': retune_speed, 'octave': octave,
                       'miku_amount': miku_amount, 'method': method}}
