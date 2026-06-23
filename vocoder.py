# -*- coding: utf-8 -*-
"""
vocoder.py -- Núcleo DSP del proyecto "Phase Vocoder (Dolson 1986) + Pedal Miku".

Procesamiento Digital de Señales e Imágenes (INFB6063) -- UTEM 2026-1
Estudiante: Francisco Alejandro Pinto Abraham -- RUT 21.571.239-7

Contiene:
  1) Utilidades de señal y análisis de Fourier REUTILIZADAS del proyecto previo
     `miku_pedal.ipynb` (declaradas explícitamente como código reutilizado).
  2) La REPRODUCCIÓN del phase vocoder de Dolson (STFT -> frecuencia instantánea por
     fase -> time-stretch -> pitch-shift).
  3) La EXTENSIÓN propia: phase vocoder con preservación de formantes (corrección de
     envolvente espectral por cepstrum).
  4) Tres métodos de pitch-shift comparables: remuestreo ingenuo (baseline),
     phase vocoder, y phase vocoder + formantes.
  5) Métricas de evaluación (error de f0 en cents, error de duración, centroide de la
     envolvente, distancia log-espectral).

Todo es DSP clásico del curso: DFT/FFT, STFT, enventanado Hann, fase, cepstrum,
overlap-add, muestreo/interpolación. Sin ML / deep learning / LLM.
"""

import os
import numpy as np

SR = 22050                       # frecuencia de muestreo de trabajo (Hz)
EPS = 1e-9

# Mapa vocal -> formantes (fc, ancho de banda, ganancia). Reutilizado de miku_pedal.ipynb.
VOWEL_FORMANTS = {
    'a': ((800.0, 90.0, 1.00), (1150.0, 110.0, 0.82), (2900.0, 240.0, 0.38)),
    'e': ((500.0, 80.0, 0.95), (1750.0, 150.0, 0.88), (2450.0, 220.0, 0.34)),
    'i': ((320.0, 70.0, 0.92), (2200.0, 170.0, 0.95), (3000.0, 260.0, 0.42)),
    'o': ((500.0, 80.0, 0.96), (900.0, 110.0, 0.82), (2600.0, 240.0, 0.32)),
    'u': ((350.0, 70.0, 0.96), (800.0, 100.0, 0.78), (2200.0, 230.0, 0.28)),
}


# =====================================================================================
# 1) UTILIDADES REUTILIZADAS de miku_pedal.ipynb  (declaradas como código reutilizado)
# =====================================================================================
def ensure_mono_float(audio):
    """Fuerza la señal a mono float32 y limpia NaN/Inf. [Reutilizado de miku_pedal.ipynb]"""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim > 1:
        a = np.mean(a, axis=-1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).flatten().astype(np.float32)


def normalize_audio(audio, peak=0.95, eps=EPS):
    """Escala la señal para que su pico no supere 'peak'. [Reutilizado de miku_pedal.ipynb]"""
    a = ensure_mono_float(audio)
    if a.size == 0:
        return a
    m = float(np.max(np.abs(a)))
    return a.copy() if m < eps else (a / m * peak).astype(np.float32)


def apply_short_fade(audio, fade=64):
    """Fade-in/out lineal para unir granos sin clicks. [Reutilizado de miku_pedal.ipynb]"""
    a = ensure_mono_float(audio)
    f = int(min(max(fade, 0), a.size // 2))
    if f <= 1:
        return a
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    a[:f] *= ramp
    a[-f:] *= ramp[::-1]
    return a


def compute_rfft(x, fs=SR):
    """DFT de señal real: devuelve frecuencias (Hz) y magnitud. [Reutilizado de miku_pedal.ipynb]"""
    x = ensure_mono_float(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), d=1.0 / fs)
    return f, np.abs(X)


def spectrum_peaks_valleys(x, fs=SR, n_peaks=6, fmin=70.0, fmax=2000.0):
    """Máximos (picos) y mínimos (valles) locales del espectro. [Reutilizado de miku_pedal.ipynb]"""
    f, mag = compute_rfft(x, fs)
    band = (f >= fmin) & (f <= fmax)
    fb, mb = f[band], mag[band]
    peaks = [i for i in range(1, len(mb) - 1) if mb[i] > mb[i - 1] and mb[i] >= mb[i + 1]]
    valleys = [i for i in range(1, len(mb) - 1) if mb[i] < mb[i - 1] and mb[i] <= mb[i + 1]]
    top = sorted(sorted(peaks, key=lambda i: -mb[i])[:n_peaks])
    return fb, mb, np.array(top, dtype=int), np.array(valleys, dtype=int)


def apply_formants(src, sr, vowel):
    """Color de vocal por máscara de formantes en frecuencia. [Reutilizado de miku_pedal.ipynb]"""
    src = ensure_mono_float(src)
    n = len(src)
    if n < 4:
        return src
    spec = np.fft.rfft(src)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    H = np.full_like(freqs, 0.05, dtype=np.float64)
    for fc, bw, g in VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS['a']):
        H += g * np.exp(-0.5 * ((freqs - fc) / max(bw, 1.0)) ** 2)
    return np.fft.irfft(spec * H, n=n).astype(np.float32)


def build_guitar_unit_db(x, sr, unit_ms=140, hop_ms=70, fmin=70.0, fmax=1000.0, gate=0.12):
    """Base de 'difonos': granos de guitarra enventanados con f0 y energía. [Reutilizado]"""
    x = ensure_mono_float(x)
    L = max(64, int(unit_ms / 1000.0 * sr))
    H = max(1, int(hop_ms / 1000.0 * sr))
    win = np.hanning(L).astype(np.float32)
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    grains = [(s, x[s:s + L] * win) for s in range(0, len(x) - L, H)]
    if not grains:
        return []
    energies = [float(np.sqrt(np.mean(g ** 2))) for _, g in grains]
    thr = max(energies) * float(gate)
    db = []
    for (s, g), e in zip(grains, energies):
        if e < thr:
            continue
        mag = np.abs(np.fft.rfft(g))
        mb = np.where(band, mag, 0.0)
        db.append({'wave': g.astype(np.float32), 'f0': float(freqs[int(np.argmax(mb))]), 'energy': e})
    return db


def track_dominant_per_frame(x, fs=SR, frame=2048, hop=1024, fmin=70.0, fmax=2000.0):
    """Frecuencia dominante y energía por tramo (STFT). [Reutilizado de miku_pedal.ipynb]"""
    x = ensure_mono_float(x)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    win = np.hanning(frame).astype(np.float32)
    freqs = np.fft.rfftfreq(frame, d=1.0 / fs)
    band = (freqs >= fmin) & (freqs <= fmax)
    times, f0, energy = [], [], []
    for s in range(0, len(x) - frame + 1, hop):
        fr = x[s:s + frame] * win
        mag = np.abs(np.fft.rfft(fr))
        mb = np.where(band, mag, 0.0)
        f0.append(float(freqs[int(np.argmax(mb))]))
        energy.append(float(np.sqrt(np.mean(fr ** 2))))
        times.append((s + frame / 2) / fs)
    return np.array(times), np.array(f0), np.array(energy)


def select_unit(db, f_target):
    """Selecciona el difono de f0 más cercana al objetivo. [Reutilizado de miku_pedal.ipynb]"""
    return min(db, key=lambda u: abs(u['f0'] - f_target))


def mix_over(dry, wet, mix=0.6):
    """Mezcla guitarra seca + voz concatenativa. [Reutilizado de miku_pedal.ipynb]"""
    dry = ensure_mono_float(dry)
    wet = ensure_mono_float(wet)
    L = max(len(dry), len(wet))
    d = np.zeros(L, dtype=np.float32); d[:len(dry)] = dry
    w = np.zeros(L, dtype=np.float32); w[:len(wet)] = wet
    return normalize_audio((1.0 - mix) * d + mix * w)


# =====================================================================================
# 2) STFT / ISTFT  (núcleo de análisis-síntesis, materia del curso: U2-L1)
# =====================================================================================
def stft(x, n_fft=1024, hop=256, win=None):
    """Qué hace: Transformada de Fourier de Tiempo Corto.
    Curso: Unit 2 - Lecture 1 (STFT/enventanado), Unit 1 - L3 (DFT).
    Matemática: X[m,k] = sum_n x[n + m*hop] w[n] e^{-j 2π k n / Nfft}. Devuelve (frames, bins).
    """
    x = ensure_mono_float(x)
    if win is None:
        win = np.hanning(n_fft).astype(np.float32)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    cols = []
    for m in range(n_frames):
        s = m * hop
        cols.append(np.fft.rfft(x[s:s + n_fft] * win))
    return np.array(cols)


def istft(X, n_fft=1024, hop=256, win=None, length=None):
    """Qué hace: reconstrucción por superposición-suma (overlap-add) con normalización de ventana.
    Curso: Unit 2 - Lecture 1/2 (enventanado y superposición).
    Matemática: x[n] = sum_m (IDFT(X[m]) w[n]) / sum_m w[n]^2  (OLA con ventana al cuadrado).
    """
    if win is None:
        win = np.hanning(n_fft).astype(np.float32)
    n_frames = X.shape[0]
    out_len = n_fft + hop * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float64)
    wsum = np.zeros(out_len, dtype=np.float64)
    for m in range(n_frames):
        frame = np.fft.irfft(X[m], n=n_fft)
        s = m * hop
        y[s:s + n_fft] += frame * win
        wsum[s:s + n_fft] += win ** 2
    y = y / np.where(wsum > EPS, wsum, 1.0)
    if length is not None:
        y = y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))
    return y.astype(np.float32)


# =====================================================================================
# 3) PHASE VOCODER  (REPRODUCCIÓN de Dolson 1986)
# =====================================================================================
def phase_vocoder(X, stretch, hop, n_fft):
    """Qué hace: estira/comprime en el tiempo una STFT preservando la frecuencia, propagando la fase
    por la FRECUENCIA INSTANTÁNEA (corazón del phase vocoder de Dolson 1986).
    Curso: Unit 2 - L1 (STFT), Unit 1 - L2/3 (fase de la DFT), enventanado.
    Matemática: para cada bin k, la fase esperada por salto es Δφ_esp = 2π·hop·k/Nfft. La desviación
    medida Δφ - Δφ_esp se envuelve a (-π,π] (heterodinaje) y da la frecuencia instantánea; la fase de
    síntesis se acumula con ese incremento. La magnitud se interpola linealmente entre tramas.

    stretch > 1 -> salida más larga (más lenta);  stretch < 1 -> más corta (más rápida).
    """
    n_frames, n_bins = X.shape
    omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft       # avance de fase esperado por bin
    Xpad = np.vstack([X, np.zeros((1, n_bins), dtype=X.dtype)])  # tramo extra para interpolar
    steps = np.arange(0, n_frames, 1.0 / stretch)               # posiciones de análisis (fraccionarias)
    out = np.zeros((len(steps), n_bins), dtype=np.complex128)
    phase_acc = np.angle(Xpad[0])
    for i, step in enumerate(steps):
        f0 = int(np.floor(step))
        frac = step - f0
        mag = (1.0 - frac) * np.abs(Xpad[f0]) + frac * np.abs(Xpad[f0 + 1])
        out[i] = mag * np.exp(1j * phase_acc)
        dphase = np.angle(Xpad[f0 + 1]) - np.angle(Xpad[f0]) - omega
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))   # envolver a (-π,π]
        phase_acc = phase_acc + omega + dphase
    return out


def time_stretch_pv(x, stretch, n_fft=1024, hop=256):
    """Qué hace: cambia la DURACIÓN sin cambiar el tono, vía phase vocoder.
    Curso: Unit 2 - L1 (STFT/OLA), Dolson 1986. stretch>1 = más largo.
    """
    x = ensure_mono_float(x)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    win = np.hanning(n_fft).astype(np.float32)
    X = stft(x, n_fft=n_fft, hop=hop, win=win)
    Xs = phase_vocoder(X, stretch, hop=hop, n_fft=n_fft)
    target = int(round(len(x) * stretch))
    return istft(Xs, n_fft=n_fft, hop=hop, win=win, length=target)


def _resample_linear(x, n_out):
    """Remuestreo por interpolación lineal a 'n_out' muestras (muestreo, U1-L3)."""
    x = ensure_mono_float(x)
    if x.size < 2 or n_out < 2:
        return x
    idx = np.linspace(0.0, x.size - 1, int(n_out))
    return np.interp(idx, np.arange(x.size), x).astype(np.float32)


def pitch_shift_pv(x, n_semitones, n_fft=1024, hop=256):
    """Qué hace: PITCH-SHIFT preservando la DURACIÓN (método de Dolson): time-stretch + remuestreo.
    Curso: Dolson 1986; Unit 2 - L1 (STFT), Unit 1 - L3 (remuestreo).
    Matemática: ratio = 2^(n/12). Se estira el tiempo por 'ratio' (queda más largo) y luego se
    remuestrea de vuelta a la longitud original -> los armónicos suben por 'ratio' y la duración se conserva.
    """
    x = ensure_mono_float(x)
    ratio = 2.0 ** (float(n_semitones) / 12.0)
    if abs(n_semitones) < 1e-6 or x.size < n_fft:
        return x
    stretched = time_stretch_pv(x, ratio, n_fft=n_fft, hop=hop)
    return _resample_linear(stretched, len(x))


def pitch_shift_resample(x, n_semitones):
    """Qué hace: PITCH-SHIFT por REMUESTREO ingenuo (BASELINE = el pitch_fit original de Miku).
    Curso: Unit 1 - L3 (muestreo/interpolación). [Equivalente al pitch_fit de miku_pedal.ipynb]
    Matemática: leer la señal a paso 'ratio' = 2^(n/12). CAMBIA la duración y CORRE los formantes.
    """
    x = ensure_mono_float(x)
    ratio = 2.0 ** (float(n_semitones) / 12.0)
    if abs(n_semitones) < 1e-6 or x.size < 4:
        return x
    pos = np.arange(0, len(x), ratio)
    if pos.size < 4:
        return x
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


# =====================================================================================
# 4) EXTENSIÓN PROPIA: preservación de formantes por envolvente cepstral
# =====================================================================================
def cepstral_envelope(mag, n_lifter=30):
    """Qué hace: estima la ENVOLVENTE espectral (formantes) suavizando el log-espectro con un lifter cepstral.
    Curso: Unit 1 - L2/3 (DFT/IDFT) + filtrado en cuefrencia (cepstrum).
    Matemática: env = exp( low-quefrency( IDFT( log|S| ) ) ). Las cuefrencias altas (estructura fina
    armónica) se ponen a cero; quedan solo los formantes.
    """
    mag = np.asarray(mag, dtype=np.float64)
    log_mag = np.log(mag + EPS)
    cep = np.fft.irfft(log_mag)                  # cepstro real, longitud 2*(bins-1)
    n = cep.size
    k = int(min(max(n_lifter, 1), n // 2 - 1))
    lifter = np.zeros(n)
    lifter[:k] = 1.0
    lifter[-k + 1:] = 1.0 if k > 1 else 0.0      # mantener simetría de cuefrencias bajas
    env = np.exp(np.fft.rfft(cep * lifter).real)
    return env[:mag.size]


def pitch_shift_pv_formant(x, n_semitones, n_fft=1024, hop=256, n_lifter=30):
    """Qué hace: phase vocoder con PRESERVACIÓN DE FORMANTES (extensión propia).
    Sube el tono con el phase vocoder y luego re-impone la envolvente espectral ORIGINAL, de modo que
    los formantes (timbre/identidad de la vocal) no se desplacen junto con los armónicos.
    Curso: Dolson 1986 + Unit 2 - L3 (filtro como máscara H[k]=env_o/env_s) + cepstrum (U1-L2/3).
    Matemática: y = IDFT( DFT(pv) · env_orig/env_pv ), con env_* envolventes cepstrales.
    """
    x = ensure_mono_float(x)
    shifted = pitch_shift_pv(x, n_semitones, n_fft=n_fft, hop=hop)
    if x.size < 8:
        return shifted
    N = max(len(x), len(shifted))
    Xo = np.fft.rfft(x, n=N)
    Xs = np.fft.rfft(shifted, n=N)
    env_o = cepstral_envelope(np.abs(Xo), n_lifter=n_lifter)
    env_s = cepstral_envelope(np.abs(Xs), n_lifter=n_lifter)
    H = env_o / (env_s + EPS)
    H = np.clip(H, 1e-3, 1e3)                     # limitar a ±60 dB (evita amplificar valles/ruido)
    y = np.fft.irfft(Xs * H, n=N)[:len(shifted)]
    return y.astype(np.float32)


# Despachador por nombre de método (usado por la síntesis y los experimentos).
PITCH_METHODS = {
    'resample': pitch_shift_resample,
    'pv': pitch_shift_pv,
    'pv_formant': pitch_shift_pv_formant,
}


# =====================================================================================
# 5) SÍNTESIS CONCATENATIVA con método de pitch-shift intercambiable
# =====================================================================================
def _pitch_to_target(wave, sr, f_src, f_target, method):
    """Lleva un grano de f_src a f_target usando el método de pitch-shift indicado."""
    wave = ensure_mono_float(wave)
    if f_src <= 1e-6 or f_target <= 1e-6 or wave.size < 4:
        return wave
    n_semi = 12.0 * np.log2(float(np.clip(f_target / f_src, 0.5, 2.0)))
    return PITCH_METHODS[method](wave, n_semi)


def concatenative_voice(x, sr, times, f0, energy, db, method='pv',
                        vowels='a e i o u', gate=0.4, overlap=0.05):
    """Qué hace: SÍNTESIS CONCATENATIVA estilo Vocaloid (reutilizada de miku_pedal.ipynb) pero con el
    método de pitch-shift INTERCAMBIABLE ('resample' | 'pv' | 'pv_formant').
    Curso: Unit 2 - L1 (seguimiento STFT), pitch-shift (Dolson), U2-L3 (formantes), U2-L2 (superposición).
    """
    n = len(ensure_mono_float(x))
    layer = np.zeros(n, dtype=np.float32)
    if not db or len(energy) == 0:
        return layer
    emax = float(np.max(energy)) + 1e-12
    active = energy > (emax * float(gate))
    vlist = [v for v in vowels.split() if v in VOWEL_FORMANTS] or ['a']
    segs, i = [], 0
    while i < len(active):
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            segs.append((i, j)); i = j
        else:
            i += 1
    wi = 0
    for a, b in segs:
        t0 = times[a]; t_end = times[min(b, len(times) - 1)]
        pos = t0; guard = 0
        while pos < t_end and guard < 4000:
            fi = min(int(np.searchsorted(times, pos)), len(f0) - 1)
            ft = float(f0[fi]); vel = float(np.clip(energy[fi] / emax, 0.45, 1.0))
            u = select_unit(db, ft)
            g = _pitch_to_target(u['wave'], sr, u['f0'], ft, method)   # pitch-shift intercambiable
            g = apply_formants(g, sr, vlist[wi % len(vlist)])
            g = apply_short_fade(normalize_audio(g, 0.9), fade=64)
            start = int(pos * sr); end = min(n, start + len(g))
            if end > start:
                layer[start:end] += g[:end - start] * vel
            pos += max(0.08, len(g) / sr - overlap)
            wi += 1; guard += 1
    return normalize_audio(layer)


# =====================================================================================
# 6) MÉTRICAS DE EVALUACIÓN
# =====================================================================================
def estimate_f0_autocorr(x, fs=SR, fmin=70.0, fmax=1000.0):
    """Qué hace: estima f0 por autocorrelación (robusto para tonos armónicos).
    Curso: Unit 1 - L1/L2 (correlación/periodicidad).
    Matemática: r[τ] = sum_n x[n]x[n+τ]; f0 = fs / argmax_τ r[τ] dentro de [1/fmax, 1/fmin].
    """
    x = ensure_mono_float(x)
    x = x - np.mean(x)
    if x.size < 8 or np.max(np.abs(x)) < EPS:
        return 0.0
    r = np.correlate(x, x, mode='full')[x.size - 1:]
    tmin = int(fs / fmax); tmax = int(fs / fmin)
    tmax = min(tmax, r.size - 1)
    if tmax <= tmin + 1:
        return 0.0
    tau = tmin + int(np.argmax(r[tmin:tmax]))
    # refinamiento parabólico
    if 0 < tau < r.size - 1:
        a, b, c = r[tau - 1], r[tau], r[tau + 1]
        denom = (a - 2 * b + c)
        if abs(denom) > EPS:
            tau = tau + 0.5 * (a - c) / denom
    return float(fs / tau) if tau > 0 else 0.0


def cents_error(f_est, f_target):
    """Error de afinación en cents: 1200·log2(f_est/f_target). 100 cents = 1 semitono."""
    if f_est <= 0 or f_target <= 0:
        return float('nan')
    return float(1200.0 * np.log2(f_est / f_target))


def duration_error_pct(n_out, n_expected):
    """Error de duración en % respecto a la duración esperada (la del original)."""
    if n_expected <= 0:
        return float('nan')
    return float(100.0 * (n_out - n_expected) / n_expected)


def spectral_centroid(x, fs=SR):
    """Centroide espectral (Hz): sum(f·|S|)/sum(|S|). Proxy del 'brillo' del timbre."""
    f, mag = compute_rfft(x, fs)
    s = float(np.sum(mag))
    return float(np.sum(f * mag) / s) if s > EPS else 0.0


def envelope_centroid(x, fs=SR, n_lifter=30):
    """Centroide de la ENVOLVENTE espectral (formantes), independiente de la posición de los armónicos.
    Es la métrica clave de preservación de formantes: si los formantes se mueven, este centroide cambia.
    """
    f, mag = compute_rfft(x, fs)
    env = cepstral_envelope(mag, n_lifter=n_lifter)
    s = float(np.sum(env))
    return float(np.sum(f * env) / s) if s > EPS else 0.0


def log_spectral_distance(x_ref, x_test, fs=SR, n_lifter=30):
    """Distancia log-espectral de FORMA entre ENVOLVENTES (dB). Mide cuánto cambió la FORMA de la
    envolvente (formantes), invariante a la ganancia global: se resta el desplazamiento medio en dB.
    Matemática: d=20·log10(env_ref/env_test); LSD = sqrt( mean( (d - mean(d))^2 ) ).
    """
    N = max(len(ensure_mono_float(x_ref)), len(ensure_mono_float(x_test)))
    er = cepstral_envelope(np.abs(np.fft.rfft(ensure_mono_float(x_ref), n=N)), n_lifter)
    et = cepstral_envelope(np.abs(np.fft.rfft(ensure_mono_float(x_test), n=N)), n_lifter)
    d = 20.0 * np.log10((er + EPS) / (et + EPS))
    d = d - np.mean(d)                      # quitar diferencia de ganancia -> sólo forma
    return float(np.sqrt(np.mean(d ** 2)))


# =====================================================================================
# 7) SEÑALES SINTÉTICAS DE PRUEBA (verdad de terreno conocida)
# =====================================================================================
def synth_vowel(f0=150.0, vowel='a', dur=1.0, fs=SR, n_harm=40, seed=2026):
    """Qué hace: genera una señal cuasi-vocálica: serie armónica de f0 modelada por una envolvente de
    formantes FIJA (la de 'vowel'). f0 y formantes son CONOCIDOS -> sirve de verdad de terreno.
    Curso: Unit 1 - L2 (suma de sinusoides/armónicos), U2-L3 (formantes).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * fs)) / fs
    formants = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS['a'])

    def envelope(fr):
        h = 0.03
        for fc, bw, g in formants:
            h += g * np.exp(-0.5 * ((fr - fc) / max(bw, 1.0)) ** 2)
        return h

    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        fk = k * f0
        if fk >= fs / 2:
            break
        x += envelope(fk) * np.sin(2.0 * np.pi * fk * t + 2.0 * np.pi * rng.random())
    x += 0.005 * rng.standard_normal(t.size)
    return normalize_audio(x)


def synth_chirp(f_start=120.0, f_end=600.0, dur=1.0, fs=SR):
    """Chirp lineal (señal no estacionaria) para estresar el seguimiento de fase. Curso: U2-L1."""
    t = np.arange(int(dur * fs)) / fs
    k = (f_end - f_start) / dur
    phase = 2.0 * np.pi * (f_start * t + 0.5 * k * t ** 2)
    return normalize_audio(np.sin(phase))


# =====================================================================================
# 8) E/S de audio (andamiaje, no es materia del curso)
# =====================================================================================
def load_audio(path, sr=SR):
    """Carga un audio a mono y lo remuestrea a 'sr'. Usa soundfile (libsndfile soporta mp3) y, si hace
    falta, remuestreo lineal. E/S/preprocesamiento (no es materia del curso)."""
    import soundfile as sf
    y, sr_in = sf.read(path, always_2d=False)
    y = ensure_mono_float(y)
    if sr_in != sr and y.size > 1:
        n_out = int(round(y.size * sr / sr_in))
        y = _resample_linear(y, n_out)
    return normalize_audio(y)
