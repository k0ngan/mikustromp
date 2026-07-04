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
# 1b) ACONDICIONADO DE LA VOZ Y REGISTRO DE MIKU
# =====================================================================================
def precondition_voice(x, sr=SR, hp_hz=70.0, declick=True):
    """Qué hace: acondiciona una voz de micrófono para el autotune. Curso: DFT/FFT (rfft/irfft).
    (1) quita DC; (2) pasa-altos por FFT (pone a cero los bins < hp_hz) para sacar rumble/sub-graves;
    (3) de-click: recorta transientes aislados (p. ej. el "pop" al empezar a grabar) por encima de un
    umbral robusto; (4) normaliza por un pico ROBUSTO (percentil 99.5), no por el máximo, de modo que una
    toma floja o con un click al inicio quede a un nivel usable sin que el click domine. Sin ML.
    """
    x = V.ensure_mono_float(x)
    if x.size < 8:
        return x
    x = x - float(np.mean(x))                             # (1) quita DC
    X = np.fft.rfft(x)                                    # (2) pasa-altos por FFT
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    X[freqs < hp_hz] = 0.0
    x = np.fft.irfft(X, n=x.size).astype(np.float64)
    if declick:                                          # (3) de-click de transientes aislados
        a = np.abs(x)
        med = float(np.median(a[a > EPS])) if np.any(a > EPS) else 0.0
        rms = float(np.sqrt(np.mean(x ** 2)))
        thr = max(20.0 * med, 8.0 * rms)
        if thr > EPS:
            x = np.clip(x, -thr, thr)
    p = float(np.percentile(np.abs(x), 99.5))            # (4) normalización robusta
    if p > EPS:
        x = np.clip(x / p * 0.95, -1.0, 1.0)
    return x.astype(np.float32)


def miku_register_octave(f0, voiced, target_f0=466.0, lo=-2, hi=3):
    """Qué hace: número ENTERO de octavas para llevar la voz al registro de Miku (~target_f0),
    conservando el contorno melódico. K = round(log2(target / mediana_f0_voiced)), acotado a [lo, hi].
    Curso: logaritmos de frecuencia.
    """
    f0 = np.asarray(f0, dtype=float)
    v = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (f0 > 0)
    if not np.any(v):
        return 0
    med = float(np.median(f0[v]))
    if med <= 0:
        return 0
    return int(np.clip(int(np.round(np.log2(target_f0 / med))), lo, hi))


def _ola_normalize(out, wsum, n):
    """Reconstrucción WOLA robusta: divide por la suma de ventanas^2 SOLO donde es significativa (evita
    que los bordes con solape mínimo -> denominador ~0 amplifiquen y creen picos)."""
    wmax = float(np.max(wsum)) + EPS
    denom = np.where(wsum[:n] > 1e-3 * wmax, wsum[:n], np.inf)   # inf -> esas muestras quedan en 0
    return out[:n] / denom


def _robust_normalize(y, peak=0.95, pct=99.9):
    """Normaliza por un pico ROBUSTO (percentil `pct`, no el máximo) y satura suave a [-1,1], de modo que
    un pico aislado no deje toda la señal inaudible (el problema del crest factor gigante)."""
    y = V.ensure_mono_float(y)
    if y.size == 0:
        return y
    r = float(np.percentile(np.abs(y), pct))
    if r < EPS:
        return y
    return np.clip(y / r * peak, -1.0, 1.0).astype(np.float32)


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
                   octave='auto', method='pv_formant', frame=2048, hop=512,
                   fmin=80.0, fmax=1000.0, snap_clip=6.0, gate=0.06,
                   precondition=True, target_f0=466.0):
    """Qué hace: AFINA una voz a una escala y la lleva al registro de Miku, corrigiendo el tono de forma
    variable en el tiempo. Curso: STFT/enventanado (Hann), phase vocoder (Dolson), overlap-add.
    Método: (opcional) se acondiciona la voz (`precondition_voice`); se sigue el tono; se enmarca con
    ventana Hann y 75% de solape; para cada cuadro se aplica `pitch_shift_pv_formant` por
    (corrección del snap acotada a ±snap_clip) + (12·K de la octava de registro), y se reconstruye por
    superposición-suma (WOLA). Los cuadros sin voz o de baja energía se **silencian** (compuerta `gate`)
    para no sintetizar ruido tonal en las pausas. La DURACIÓN se conserva.

    octave: 'auto' lleva la voz al registro de Miku (~target_f0); un entero fija la octava; 0 = sin subir.
    Devuelve (audio_afinado, info) con info = contornos y la octava K aplicada.
    """
    x = V.ensure_mono_float(x)
    if precondition:
        x = precondition_voice(x, sr)
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    win = np.hanning(frame).astype(np.float64)

    # contorno de tono; corrección del snap SIN la octava (se maneja aparte para no recortarla)
    times, f0, voiced = track_f0_pyin(x, sr, fmin=fmin, fmax=fmax, frame=frame, hop=hop)
    key_pc = note_to_pitch_class(key)
    scale_set = set(SCALES.get(scale, SCALES['chromatic']))
    corr_snap, orig_midi, snapped, target = correction_semitones(
        f0, voiced, key_pc, scale_set, strength, retune_speed, octave=0)

    # registro: octava global entera (automática al registro de Miku, o entero fijo, o 0)
    if isinstance(octave, str) and octave.lower() == 'auto':
        K = miku_register_octave(f0, voiced, target_f0=target_f0)
    else:
        try:
            K = int(octave)
        except (TypeError, ValueError):
            K = 0
    oct_semi = 12.0 * K

    # energía por cuadro (para la compuerta de silencios)
    n_frames = 1 + (x.size - frame) // hop
    frame_e = np.array([np.sqrt(np.mean(x[i * hop:i * hop + frame] ** 2)) for i in range(n_frames)])
    emax = float(frame_e.max()) + EPS

    out = np.zeros(x.size + frame, dtype=np.float64)
    wsum = np.zeros_like(out)
    tv = np.asarray(times)
    applied = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * hop
        tc = (s + frame / 2.0) / sr                 # centro del cuadro en segundos
        j = int(np.argmin(np.abs(tv - tc))) if tv.size else 0
        voiced_j = bool(voiced[j]) if j < voiced.size else False
        if (not voiced_j) or frame_e[i] < gate * emax:   # compuerta: silencio/no-voz -> no sintetizar
            continue
        snap = float(corr_snap[j]) if j < corr_snap.size else 0.0
        n_semi = float(np.clip(snap, -snap_clip, snap_clip)) + oct_semi
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
    y = _ola_normalize(out, wsum, x.size)

    info = {'times': times, 'f0': f0, 'voiced': voiced, 'orig_midi': orig_midi,
            'snapped_midi': snapped, 'target_midi': target + oct_semi, 'corr': corr_snap,
            'applied': applied, 'octave_K': K}
    return _robust_normalize(y), info


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
    e_gate = 1e-3 * (float(np.max(np.abs(x))) + EPS)     # umbral de silencio (relativo al pico)
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s + frame] * win
        if np.sqrt(np.mean(seg ** 2)) < e_gate:          # silencio: no imponer formantes sobre ruido
            out[s:s + frame] += seg
            wsum[s:s + frame] += win ** 2
            continue
        spec = np.fft.rfft(seg)
        env = V.cepstral_envelope(np.abs(spec), n_lifter)
        env = env / (np.mean(env) + EPS)
        H = np.clip(tmpl_i / (env + EPS), 0.06, 16.0) ** amount   # +-24 dB (evita picos por bins débiles)
        y = np.fft.irfft(spec * H, n=frame)
        out[s:s + frame] += y * win
        wsum[s:s + frame] += win ** 2
    y = _ola_normalize(out, wsum, x.size)
    return _robust_normalize(y)


# =====================================================================================
# 5) PIPELINE COMPLETO  (autotune + timbre Miku)
# =====================================================================================
def mikutune(x, sr=SR, key='C', scale='major', strength=1.0, retune_speed=1.0, octave='auto',
             miku_amount=0.9, method='pv_formant', ref_miku=None, frame=2048, hop=512,
             fmin=80.0, fmax=1000.0, gate=0.06, precondition=True, target_f0=466.0):
    """Qué hace: pipeline completo del "Autotune Miku". Acondiciona la voz, la afina a la escala, la lleva
    al registro de Miku (octava automática) y le transfiere el timbre de Miku. Devuelve un dict con la
    entrada, la etapa afinada, la salida con voz Miku, la plantilla de formantes, los contornos y los
    parámetros usados (incluida la octava K aplicada).
    """
    x = V.ensure_mono_float(x)
    auto, info = autotune_voice(x, sr, key=key, scale=scale, strength=strength,
                                retune_speed=retune_speed, octave=octave, method=method,
                                frame=frame, hop=hop, fmin=fmin, fmax=fmax,
                                gate=gate, precondition=precondition, target_f0=target_f0)
    template = None
    miku = auto
    if ref_miku is not None and miku_amount > 0:
        template = miku_formant_template(ref_miku, sr, n_fft=frame, hop=frame // 2)
        miku = miku_formant_transfer(auto, sr, template=template, amount=miku_amount,
                                     frame=frame, hop=hop)
    return {'input': x, 'auto': auto, 'miku': miku, 'template': template, 'info': info,
            'params': {'key': key, 'scale': scale, 'strength': strength,
                       'retune_speed': retune_speed, 'octave': octave,
                       'octave_K': info.get('octave_K', 0),
                       'miku_amount': miku_amount, 'method': method}}
