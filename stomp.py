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


def _odd(k, n):
    """Kernel impar valido para medfilt (1 <= k <= n, impar)."""
    k = int(min(max(k, 1), n if n % 2 == 1 else n - 1))
    return k if k % 2 == 1 else max(1, k - 1)


def _postprocess_f0(f0, voiced, med=5):
    """Qué hace: limpia el contorno de f0 quitando saltos de octava espurios (pYIN se engancha a
    armónicos) y suavizándolo con un filtro de mediana, para que el glissando sea coherente.
    Matemática: repara cada valor que esté ~2x o ~0.5x de la mediana local; luego mediana movil.
    """
    f = np.array(f0, dtype=float)
    v = np.asarray(voiced) & np.isfinite(f)
    idx = np.where(v)[0]
    if idx.size < 5:
        return f
    try:
        from scipy.signal import medfilt
        fv = f[idx]
        base = medfilt(fv, kernel_size=_odd(9, fv.size))      # centro local robusto
        ratio = fv / np.where(base > EPS, base, 1.0)
        fv = np.where(ratio > 1.8, fv / 2.0, fv)              # octava arriba espuria
        fv = np.where(ratio < 0.55, fv * 2.0, fv)             # octava abajo espuria
        fv = medfilt(fv, kernel_size=_odd(med, fv.size))      # suavizado
        f[idx] = fv
    except Exception:
        pass
    return f


# ------------------------------------------------------------------ detección de notas
def track_f0_pyin(x, sr=SR, fmin=70.0, fmax=600.0, frame=2048, hop=512, clean=True):
    """Qué hace: estima el contorno de f0 cuadro a cuadro y marca tramos sonoros (voiced).
    Curso: detección de tono monofónica (YIN/autocorrelación); STFT/enventanado.
    Usa librosa.pyin (DSP clásico); si no está, cae a la autocorrelación por tramos de vocoder.
    fmax=600 evita que pYIN se enganche a armónicos de la guitarra (errores de octava). Si clean,
    repara saltos de octava y suaviza el contorno. Devuelve (times, f0, voiced) con f0=NaN sin voz.
    """
    x = V.ensure_mono_float(x)
    try:
        import librosa
        f0, vflag, vprob = librosa.pyin(x.astype(float), fmin=fmin, fmax=fmax, sr=sr,
                                        frame_length=frame, hop_length=hop)
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        voiced = np.nan_to_num(vflag, nan=0.0).astype(bool)
        f0 = np.asarray(f0, dtype=float)
        if clean:
            f0 = _postprocess_f0(f0, voiced)
        return np.asarray(times), f0, voiced
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
        times, f0, voiced = np.array(times), np.array(f0, dtype=float), np.array(voiced, dtype=bool)
        if clean:
            f0 = _postprocess_f0(f0, voiced)
        return times, f0, voiced


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
def prep_voice_grain(sample, sr=SR, target_ms=350.0, fmin=120.0, fmax=500.0):
    """Qué hace: de una muestra de voz elige la ventana MÁS ESTABLE EN TONO (no la de mayor energía)
    y estima su f0 base con la mediana de pYIN. Un grano de tono estable es clave para que el
    desplazamiento de tono sea preciso. Curso: enventanado, detección de tono.
    Devuelve (grano, f0_base).
    """
    a = V.normalize_audio(sample)
    L = int(target_ms / 1000.0 * sr)
    if a.size <= L:
        grain = V.apply_short_fade(V.normalize_audio(a), fade=int(0.012 * sr))
        f0 = V.estimate_f0_autocorr(grain, sr, fmin, fmax)
        return grain.astype(np.float32), float(f0 if fmin <= f0 <= fmax else 220.0)
    try:
        import librosa
        hop = 256
        f0, vf, vp = librosa.pyin(a.astype(float), fmin=fmin, fmax=fmax, sr=sr,
                                  frame_length=2048, hop_length=hop)
        nf = max(1, L // hop)
        best = None                                   # (std_semitonos, start, f0_mediana)
        step = max(1, L // 8)
        for s in range(0, a.size - L, step):
            i0, i1 = s // hop, s // hop + nf
            seg = f0[i0:i1]
            segv = seg[np.isfinite(seg)]
            if segv.size < 0.6 * nf:                  # exigir mayoría voiced
                continue
            std = float(np.std(12.0 * np.log2(segv / np.median(segv))))   # estabilidad en semitonos
            if best is None or std < best[0]:
                best = (std, s, float(np.median(segv)))
        if best is not None:
            _, s, vf0 = best
            grain = V.apply_short_fade(V.normalize_audio(a[s:s + L]), fade=int(0.012 * sr))
            return grain.astype(np.float32), float(vf0)
    except Exception as exc:
        print('[stomp] prep_voice_grain sin pYIN (%s); uso ventana de energía.' % str(exc)[:50])
    # respaldo: ventana de mayor energía + autocorrelación
    hop = max(1, L // 4)
    best_s, best_e = 0, -1.0
    for s in range(0, a.size - L, hop):
        e = float(np.mean(a[s:s + L] ** 2))
        if e > best_e:
            best_e, best_s = e, s
    grain = V.apply_short_fade(V.normalize_audio(a[best_s:best_s + L]), fade=int(0.012 * sr))
    f0 = V.estimate_f0_autocorr(grain, sr, fmin, fmax)
    return grain.astype(np.float32), float(f0 if fmin <= f0 <= fmax else 220.0)


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


def compute_global_octave(f0, voiced, voice_f0):
    """Offset GLOBAL de octavas (entero) para llevar la melodía de la guitarra al registro de la voz
    conservando el contorno: K = round( mediana(12*log2(f0/voice_f0)) / 12 ).
    """
    v = np.asarray(voiced) & np.isfinite(f0)
    if not np.any(v):
        return 0
    semis = 12.0 * np.log2(np.asarray(f0)[v] / max(voice_f0, EPS))
    return int(np.round(np.median(semis) / 12.0))


def _octave_clamp(n_semi, limit=12.0):
    """Pliega por octavas SOLO si el desplazamiento excede +-limit (para notas fuera del registro de la
    voz). Las notas dentro de rango quedan intactas (se conserva el contorno); las imposibles (p. ej.
    un bajo 22 semitonos abajo) suben octavas hasta poder cantarse."""
    while n_semi > limit:
        n_semi -= 12.0
    while n_semi < -limit:
        n_semi += 12.0
    return n_semi


def miku_stomp_glide(times, f0, voiced, sr, voice_grain, voice_f0, total_len,
                     method='pv_formant', frame=2048, hop=1024, range_limit=12.0):
    """Qué hace: NÚCLEO del pedal (modo glissando). La voz SIGUE EL CONTORNO de f0 de la guitarra de
    forma continua: por cada frame con solape (Hann + overlap-add) afina el grano de voz al f0 local,
    transponiendo TODO por un offset global de octavas (conserva el contorno melódico) y plegando por
    octavas solo las notas fuera del registro cantable (+-range_limit). Frames sin voz -> silencio.
    Curso: STFT/enventanado, pitch-shift, overlap-add.
    """
    out = np.zeros(int(total_len), dtype=np.float64)
    tv, fv, vv = np.asarray(times), np.asarray(f0, dtype=float), np.asarray(voiced)
    valid = vv & np.isfinite(fv)
    if not np.any(valid):
        return V.normalize_audio(out)
    K = compute_global_octave(fv, vv, voice_f0)
    win = np.hanning(frame).astype(np.float64)
    cache = {}                                            # n_semi redondeado -> grano afinado
    nframes = 1 + max(0, (int(total_len) - frame)) // hop
    for i in range(nframes):
        s = i * hop
        tc = (s + frame / 2.0) / sr
        j = int(np.argmin(np.abs(tv - tc)))
        if not valid[j]:
            continue
        n_semi = 12.0 * np.log2(max(fv[j], EPS) / max(voice_f0, EPS)) - 12.0 * K
        n_semi = _octave_clamp(float(n_semi), range_limit)    # solo pliega los extremos
        key = round(n_semi * 4) / 4.0                     # cache a 1/4 de semitono
        pitched = cache.get(key)
        if pitched is None:
            pitched = _tile_to_length(V.PITCH_METHODS[method](voice_grain, key), frame, int(0.01 * sr))
            cache[key] = pitched
        e = min(out.size, s + frame)
        L = e - s
        out[s:e] += pitched[:L] * win[:L]
    return V.normalize_audio(out)


def run_stomp(guitar, sr, voice_grain, voice_f0, method='pv_formant', mix=0.0,
              fmin=70.0, fmax=600.0, glide=True):
    """Conveniencia: corre todo el pipeline (pYIN -> contorno -> síntesis) y opcionalmente mezcla con
    la guitarra seca (mix>0 = perilla MIX). Por defecto usa el modo glissando (sigue el contorno).
    Devuelve dict con voz, mezcla, notas (para graficar) y contorno de f0.
    """
    guitar = V.ensure_mono_float(guitar)
    times, f0, voiced = track_f0_pyin(guitar, sr, fmin, fmax)
    notes = segment_notes(times, f0, voiced)
    if glide:
        wet = miku_stomp_glide(times, f0, voiced, sr, voice_grain, voice_f0, len(guitar), method=method)
    else:
        wet = miku_stomp(notes, sr, voice_grain, voice_f0, len(guitar), method=method)
    mixed = V.mix_over(guitar, wet, mix=mix) if mix > 0 else wet
    return {'times': times, 'f0': f0, 'voiced': voiced, 'notes': notes,
            'wet': wet, 'mixed': mixed}
