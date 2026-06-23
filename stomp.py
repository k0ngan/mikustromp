# -*- coding: utf-8 -*-
"""
stomp.py -- "Miku Stomp digital": el flujo del pedal Korg Miku Stomp en DSP puro.

Audio de guitarra -> detectar la nota (pYIN, monofónico) -> afinar una muestra de voz a esa nota con
el phase vocoder (preservando formantes) -> concatenar -> mezclar con la guitarra.

Reutiliza el núcleo DSP de `vocoder.py` (phase vocoder, pitch-shift, formantes). El único método de
detección de tono que se apoya en librosa es pYIN, que es un algoritmo CLÁSICO (YIN probabilístico),
no una red neuronal. Sin ML / deep learning / LLM.
"""

import numpy as np
import vocoder as V

SR = V.SR
EPS = 1e-9


# ------------------------------------------------------------------ detección de notas
def track_f0_pyin(x, sr=SR, fmin=70.0, fmax=1000.0, frame=2048, hop=512):
    """Qué hace: estima el contorno de f0 cuadro a cuadro y marca tramos sonoros (voiced).
    Curso: detección de tono monofónica (YIN/autocorrelación); STFT/enventanado.
    Usa librosa.pyin (DSP clásico); si no está, cae a la autocorrelación por tramos de vocoder.
    Devuelve (times, f0, voiced) con f0=NaN donde no hay voz.
    """
    x = V.ensure_mono_float(x)
    try:
        import librosa
        f0, vflag, vprob = librosa.pyin(x.astype(float), fmin=fmin, fmax=fmax, sr=sr,
                                        frame_length=frame, hop_length=hop)
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        voiced = np.nan_to_num(vflag, nan=0.0).astype(bool)
        return np.asarray(times), np.asarray(f0, dtype=float), voiced
    except Exception as exc:                       # respaldo sin librosa
        print('[stomp] pYIN no disponible (%s); uso autocorrelación por tramos.' % str(exc)[:60])
        win = np.hanning(frame).astype(np.float32)
        times, f0, voiced = [], [], []
        emax = 0.0
        frames = []
        for s in range(0, max(1, len(x) - frame + 1), hop):
            fr = x[s:s + frame] * win
            e = float(np.sqrt(np.mean(fr ** 2)))
            emax = max(emax, e)
            frames.append((s, fr, e))
        for s, fr, e in frames:
            fest = V.estimate_f0_autocorr(fr, sr, fmin, fmax)
            ok = (e > 0.08 * emax) and (fmin <= fest <= fmax)
            times.append((s + frame / 2) / sr)
            f0.append(fest if ok else np.nan)
            voiced.append(bool(ok))
        return np.array(times), np.array(f0, dtype=float), np.array(voiced, dtype=bool)


def segment_notes(times, f0, voiced, min_note_ms=80.0, split_semitone=0.7):
    """Qué hace: agrupa los cuadros sonoros en NOTAS (start, end, f0). Una nota se corta cuando el
    tono se aleja > split_semitone de la mediana del tramo actual. Curso: comparación de frecuencias.
    """
    notes = []
    N = len(f0)
    dt = float(times[1] - times[0]) if N > 1 else 0.02
    i = 0
    while i < N:
        if not voiced[i] or not np.isfinite(f0[i]):
            i += 1
            continue
        j = i
        vals = [f0[i]]
        while (j + 1 < N and voiced[j + 1] and np.isfinite(f0[j + 1])
               and abs(12.0 * np.log2(f0[j + 1] / np.median(vals))) < split_semitone):
            j += 1
            vals.append(f0[j])
        t0, t1 = float(times[i]), float(times[j] + dt)
        if (t1 - t0) * 1000.0 >= min_note_ms:
            notes.append((t0, t1, float(np.median(vals))))
        i = j + 1
    return notes


# ------------------------------------------------------------------ preparación de la voz
def prep_voice_grain(sample, sr=SR, target_ms=400.0, fmin=80.0, fmax=600.0):
    """Qué hace: de una muestra de voz toma un segmento estable (mayor energía) y estima su f0 base.
    Devuelve (grano, f0_base). Curso: enventanado, energía, detección de tono.
    """
    a = V.normalize_audio(sample)
    L = int(target_ms / 1000.0 * sr)
    if a.size > L:
        # ventana de mayor energía
        hop = max(1, L // 4)
        best_s, best_e = 0, -1.0
        for s in range(0, a.size - L, hop):
            e = float(np.mean(a[s:s + L] ** 2))
            if e > best_e:
                best_e, best_s = e, s
        grain = a[best_s:best_s + L]
    else:
        grain = a
    grain = V.apply_short_fade(V.normalize_audio(grain), fade=int(0.01 * sr))
    f0 = V.estimate_f0_autocorr(grain, sr, fmin, fmax)
    if not (fmin <= f0 <= fmax):
        f0 = 200.0
    return grain.astype(np.float32), float(f0)


# ------------------------------------------------------------------ síntesis del stomp
def _tile_to_length(g, n, xfade):
    """Repite el grano hasta cubrir n muestras, con crossfade en las uniones (sostiene la vocal)."""
    g = V.ensure_mono_float(g).astype(np.float64)
    Lg = g.size
    n = int(n)
    if Lg == 0 or n <= 0:
        return np.zeros(max(n, 0), dtype=np.float64)
    if Lg >= n:
        return g[:n]
    xf = int(min(xfade, Lg // 3))
    out = np.zeros(n + Lg, dtype=np.float64)
    step = max(1, Lg - xf)
    rin = np.linspace(0.0, 1.0, xf) if xf > 0 else None
    pos = 0
    while pos < n:
        seg = g.copy()
        if pos > 0 and xf > 0:
            seg[:xf] *= rin
            out[pos:pos + xf] *= (1.0 - rin)
        out[pos:pos + Lg] += seg
        pos += step
    return out[:n]


def octave_fold(n_semi, lo=-7.0, hi=7.0):
    """Pliega el desplazamiento a la octava más cercana para mantener la voz en un registro cantable
    (|n_semi| pequeño => mejor calidad del phase vocoder)."""
    n = n_semi - 12.0 * np.round(n_semi / 12.0)
    return float(np.clip(n, lo, hi))


def miku_stomp(notes, sr, voice_grain, voice_f0, total_len,
               method='pv_formant', xfade_ms=20.0, fold=True):
    """Qué hace: NÚCLEO del pedal. Para cada nota detectada, afina la muestra de voz a esa nota con el
    método elegido ('resample' robótico / 'pv' / 'pv_formant' limpio), la sostiene la duración de la
    nota (tiling con crossfade) y la coloca en el tiempo. Curso: pitch-shift, overlap-add, enventanado.
    Devuelve la 'voz húmeda' (sin mezclar con la guitarra).
    """
    out = np.zeros(int(total_len), dtype=np.float64)
    xf = int(xfade_ms / 1000.0 * sr)
    for (t0, t1, f0n) in notes:
        n_semi = 12.0 * np.log2(max(f0n, EPS) / max(voice_f0, EPS))
        n_semi = octave_fold(n_semi) if fold else float(np.clip(n_semi, -12, 12))
        pitched = V.PITCH_METHODS[method](voice_grain, n_semi)
        L = int((t1 - t0) * sr)
        if L < 8 or pitched.size < 4:
            continue
        seg = _tile_to_length(pitched, L, xf)
        seg = V.apply_short_fade(V.normalize_audio(seg, 0.9), fade=max(8, xf))
        s = int(t0 * sr)
        e = min(out.size, s + seg.size)
        if e > s:
            out[s:e] += seg[:e - s]
    return V.normalize_audio(out)


def run_stomp(guitar, sr, voice_grain, voice_f0, method='pv_formant', mix=0.0,
              fmin=70.0, fmax=1000.0):
    """Conveniencia: corre todo el pipeline (pYIN -> notas -> síntesis) y opcionalmente mezcla con la
    guitarra seca (mix>0 = perilla MIX). Devuelve dict con voz, mezcla, notas y contorno de f0.
    """
    guitar = V.ensure_mono_float(guitar)
    times, f0, voiced = track_f0_pyin(guitar, sr, fmin, fmax)
    notes = segment_notes(times, f0, voiced)
    wet = miku_stomp(notes, sr, voice_grain, voice_f0, len(guitar), method=method)
    mixed = V.mix_over(guitar, wet, mix=mix) if mix > 0 else wet
    return {'times': times, 'f0': f0, 'voiced': voiced, 'notes': notes,
            'wet': wet, 'mixed': mixed}
