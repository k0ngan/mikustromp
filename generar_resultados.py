# -*- coding: utf-8 -*-
"""
generar_resultados.py -- Diseño experimental y generación de resultados.

Compara tres métodos de pitch-shift dentro del sintetizador concatenativo "Pedal Miku":
  (A) remuestreo ingenuo  (baseline = pitch_fit original)
  (B) phase vocoder       (REPRODUCCIÓN de Dolson 1986)
  (C) phase vocoder + formantes (EXTENSIÓN propia)

Produce:
  figuras/*.png      -> todas las figuras del informe/notebook
  outputs/*.wav      -> audios sintetizados (uno por método) + mezcla
  resultados.json    -> métricas cuantitativas (reproducibles, semilla fija)

Métricas:
  - error de afinación (cents)        -> ¿el tono quedó donde se pidió?  (tono plano: pico dominante)
  - error de duración (%)             -> ¿se conservó el largo?          (resample debería fallar)
  - corrimiento de formantes (F1, centroide, LSD) -> ¿se conservó el timbre? (vocal sintética)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

import vocoder as V

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figuras')
OUT = os.path.join(HERE, 'outputs')
DATA = os.path.join(HERE, 'data')
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SR = V.SR
SEED = 2026
np.random.seed(SEED)

SHIFTS = [-7, -5, -3, +3, +5, +7, +12]
METHODS = ['resample', 'pv', 'pv_formant']
METHOD_LABEL = {'resample': 'Remuestreo (baseline)', 'pv': 'Phase vocoder', 'pv_formant': 'PV + formantes'}
METHOD_COLOR = {'resample': 'tab:red', 'pv': 'tab:blue', 'pv_formant': 'tab:green'}


# ----------------------------------------------------------------------------- helpers
def dominant_freq(x, fs=SR, fmin=60.0, fmax=1500.0):
    """f0 por pico dominante (fiable en tonos de espectro plano)."""
    f, m = V.compute_rfft(x, fs)
    b = (f >= fmin) & (f <= fmax)
    return float(f[b][np.argmax(m[b])]) if np.any(b) else 0.0


def flat_tone(f0, dur=0.8, fs=SR, n_harm=30):
    """Tono armónico de envolvente plana (tipo diente de sierra): el fundamental domina -> f0 medible."""
    t = np.arange(int(dur * fs)) / fs
    x = np.zeros_like(t)
    for k in range(1, n_harm + 1):
        if k * f0 >= fs / 2:
            break
        x += (1.0 / k) * np.sin(2.0 * np.pi * k * f0 * t)
    return V.normalize_audio(x)


def formant_peak(x, fs=SR, lo=150.0, hi=3500.0, lifter=30):
    """Ubicación del primer formante = pico de la envolvente cepstral."""
    f, mag = V.compute_rfft(x, fs)
    env = V.cepstral_envelope(mag, lifter)
    b = (f >= lo) & (f <= hi)
    return float(f[b][np.argmax(env[b])])


def spectrogram_db(x, n_fft=1024, hop=256):
    """Espectrograma en dB usando la STFT propia (U2-L1)."""
    X = V.stft(x, n_fft=n_fft, hop=hop)
    S = 20.0 * np.log10(np.abs(X).T + 1e-6)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    times = np.arange(X.shape[0]) * hop / SR
    return times, freqs, S


def plot_spectro(ax, x, title, fmax=3500):
    t, f, S = spectrogram_db(x)
    m = f <= fmax
    ax.pcolormesh(t, f[m], S[m], shading='auto', cmap='magma', vmin=S.max() - 70, vmax=S.max())
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('tiempo [s]'); ax.set_ylabel('Hz')


results = {'config': {'sr': SR, 'seed': SEED, 'shifts': SHIFTS, 'methods': METHODS,
                      'n_fft': 1024, 'hop': 256},
           'articulo': 'Dolson (1986), The Phase Vocoder: A Tutorial, Computer Music Journal 10(4).'}


# ============================================================================= FIG 1
# Reproducción: el phase vocoder cambia la DURACIÓN sin cambiar el TONO.
def fig_reproduccion():
    x = flat_tone(220.0, dur=0.6)
    slow = V.time_stretch_pv(x, 1.5)     # 50% más largo
    fast = V.time_stretch_pv(x, 0.7)     # más corto
    f_x = dominant_freq(x); f_s = dominant_freq(slow); f_f = dominant_freq(fast)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for a, sig, ttl in [(ax[0, 0], x, 'Original (%.2fs, f0=%.0f Hz)' % (len(x) / SR, f_x)),
                        (ax[0, 1], slow, 'Time-stretch x1.5 (%.2fs, f0=%.0f Hz)' % (len(slow) / SR, f_s))]:
        a.plot(np.arange(len(sig)) / SR, sig, lw=0.5)
        a.set_title(ttl, fontsize=10); a.set_xlabel('s'); a.grid(alpha=0.3)
        a.set_xlim(0, max(len(slow), len(x)) / SR)
    plot_spectro(ax[1, 0], x, 'Espectrograma original')
    plot_spectro(ax[1, 1], slow, 'Espectrograma estirado (mismas frecuencias)')
    fig.suptitle('Reproducción phase vocoder (Dolson 1986): la duración cambia, el tono NO',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig01_reproduccion_pv.png'), dpi=130); plt.close(fig)
    results['reproduccion'] = {'f0_original': f_x, 'f0_stretch_x1.5': f_s, 'f0_stretch_x0.7': f_f,
                               'dur_original_s': len(x) / SR, 'dur_x1.5_s': len(slow) / SR,
                               'dur_x0.7_s': len(fast) / SR}
    print('FIG1 reproduccion: f0 %.1f -> stretch1.5 %.1f / stretch0.7 %.1f (Hz, debe mantenerse)' %
          (f_x, f_s, f_f))


# ============================================================================= FIG 2 + tabla A
# Exactitud de tono y de duración (tono plano).
def exp_tono_duracion():
    F0S = [110.0, 165.0, 220.0, 330.0]
    rows = []
    for f0 in F0S:
        x = flat_tone(f0)
        for n in SHIFTS:
            target = f0 * 2 ** (n / 12.0)
            for m in METHODS:
                y = V.PITCH_METHODS[m](x, n)
                fe = dominant_freq(y)
                rows.append({'f0': f0, 'shift': n, 'method': m,
                             'cents_err': V.cents_error(fe, target),
                             'dur_err_pct': V.duration_error_pct(len(y), len(x))})
    results['tono_duracion'] = rows

    # resumen por método (promedio sobre f0 y shift)
    summ = {}
    for m in METHODS:
        sub = [r for r in rows if r['method'] == m]
        summ[m] = {'cents_abs_mean': float(np.mean([abs(r['cents_err']) for r in sub])),
                   'cents_abs_max': float(np.max([abs(r['cents_err']) for r in sub])),
                   'dur_abs_mean': float(np.mean([abs(r['dur_err_pct']) for r in sub])),
                   'dur_abs_max': float(np.max([abs(r['dur_err_pct']) for r in sub]))}
    results['tono_duracion_resumen'] = summ

    # figura: error de duración por shift (promedio sobre f0) + error de cents
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in METHODS:
        durs = [np.mean([r['dur_err_pct'] for r in rows if r['method'] == m and r['shift'] == n]) for n in SHIFTS]
        cents = [np.mean([abs(r['cents_err']) for r in rows if r['method'] == m and r['shift'] == n]) for n in SHIFTS]
        ax[0].plot(SHIFTS, durs, 'o-', color=METHOD_COLOR[m], label=METHOD_LABEL[m])
        ax[1].plot(SHIFTS, cents, 'o-', color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax[0].axhline(0, color='k', lw=0.7); ax[0].set_title('Error de duración (%) vs semitonos')
    ax[0].set_xlabel('semitonos'); ax[0].set_ylabel('error duración [%]'); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_title('Error de afinación |cents| vs semitonos'); ax[1].set_xlabel('semitonos')
    ax[1].set_ylabel('|error| [cents]'); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.suptitle('Tono y duración: el remuestreo cambia el largo; el phase vocoder lo conserva',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig02_tono_duracion.png'), dpi=130); plt.close(fig)
    print('TONO/DURACION resumen:')
    for m in METHODS:
        s = summ[m]
        print('  %-22s cents|mean=%.1f max=%.1f|  dur%%|mean=%.1f max=%.1f|' %
              (METHOD_LABEL[m], s['cents_abs_mean'], s['cents_abs_max'], s['dur_abs_mean'], s['dur_abs_max']))


# ============================================================================= FIG 3+4 + tabla B
# Preservación de formantes (vocal sintética con F1/F2/F3 conocidos).
def exp_formantes():
    BASES = [120.0, 150.0]
    rows = []
    for base in BASES:
        x = V.synth_vowel(base, 'a', dur=0.8, seed=SEED)
        orig_F1 = formant_peak(x); orig_centroid = V.spectral_centroid(x)
        for n in SHIFTS:
            for m in METHODS:
                y = V.PITCH_METHODS[m](x, n)
                rows.append({'base': base, 'shift': n, 'method': m,
                             'F1': formant_peak(y), 'orig_F1': orig_F1,
                             'env_centroid': V.envelope_centroid(y), 'orig_env_centroid': V.envelope_centroid(x),
                             'lsd_db': V.log_spectral_distance(x, y)})
    results['formantes'] = rows
    summ = {}
    for m in METHODS:
        sub = [r for r in rows if r['method'] == m]
        summ[m] = {'F1_abs_shift_mean': float(np.mean([abs(r['F1'] - r['orig_F1']) for r in sub])),
                   'lsd_mean_db': float(np.mean([r['lsd_db'] for r in sub])),
                   'env_centroid_ratio_mean': float(np.mean([r['env_centroid'] / max(r['orig_env_centroid'], 1e-6) for r in sub]))}
    results['formantes_resumen'] = summ

    # FIG3: envolventes a un shift fijo (+5) mostrando el formante fijo en pv_formant
    base = 150.0; n = 5
    x = V.synth_vowel(base, 'a', dur=0.8, seed=SEED)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    f, mag = V.compute_rfft(x); env_o = V.cepstral_envelope(mag, 30)
    norm = lambda e: e / np.max(e)          # normalizar para comparar FORMA (no ganancia)
    ax[0].plot(f, norm(env_o), 'k', lw=2.2, label='original (F1=%.0f Hz)' % formant_peak(x))
    for m in METHODS:
        y = V.PITCH_METHODS[m](x, n)
        fy, magy = V.compute_rfft(y); envy = V.cepstral_envelope(magy, 30)
        ax[0].plot(fy, norm(envy), color=METHOD_COLOR[m], lw=1.4, alpha=0.9,
                   label='%s (F1=%.0f)' % (METHOD_LABEL[m], formant_peak(y)))
    ax[0].set_xlim(0, 3500); ax[0].set_title('Envolvente espectral (normalizada) tras +5 semitonos (vocal /a/)')
    ax[0].set_xlabel('Hz'); ax[0].set_ylabel('envolvente norm.'); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # FIG3b: F1 vs shift
    for m in METHODS:
        f1s = [np.mean([r['F1'] for r in rows if r['method'] == m and r['shift'] == nn]) for nn in SHIFTS]
        ax[1].plot(SHIFTS, f1s, 'o-', color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    of1 = np.mean([r['orig_F1'] for r in rows])
    ax[1].axhline(of1, color='k', ls='--', lw=1, label='F1 original (%.0f Hz)' % of1)
    ax[1].set_title('Posición del formante F1 vs semitonos'); ax[1].set_xlabel('semitonos')
    ax[1].set_ylabel('F1 [Hz]'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle('Formantes: sólo "PV + formantes" mantiene el timbre al cambiar el tono',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig03_formantes.png'), dpi=130); plt.close(fig)

    # FIG4: LSD (distancia log-espectral de la envolvente) por shift
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m in METHODS:
        lsd = [np.mean([r['lsd_db'] for r in rows if r['method'] == m and r['shift'] == nn]) for nn in SHIFTS]
        ax.plot(SHIFTS, lsd, 'o-', color=METHOD_COLOR[m], label=METHOD_LABEL[m])
    ax.set_title('Distancia log-espectral de la envolvente vs original\n(menor = formantes mejor preservados)')
    ax.set_xlabel('semitonos'); ax.set_ylabel('LSD [dB]'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig04_lsd.png'), dpi=130); plt.close(fig)
    print('FORMANTES resumen:')
    for m in METHODS:
        s = summ[m]
        print('  %-22s |F1 shift|mean=%.0f Hz  LSD mean=%.1f dB  env-centroid ratio=%.2f' %
              (METHOD_LABEL[m], s['F1_abs_shift_mean'], s['lsd_mean_db'], s['env_centroid_ratio_mean']))


# ============================================================================= FIG 5
# Caso real: un grano de guitarra desplazado por los tres métodos (espectrogramas).
def fig_grano_real(grain):
    n = 7
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    plot_spectro(ax[0, 0], grain, 'Grano de guitarra original')
    for a, m in zip([ax[0, 1], ax[1, 0], ax[1, 1]], METHODS):
        y = V.PITCH_METHODS[m](grain, n)
        plot_spectro(a, y, '%s (+%d semitonos)  dur=%.2fs' % (METHOD_LABEL[m], n, len(y) / SR))
    fig.suptitle('Grano real de guitarra desplazado +7 semitonos por método', fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig05_grano_real.png'), dpi=130); plt.close(fig)
    print('FIG5 grano real listo')


# ============================================================================= FIG 6+7
# Integración: la voz concatenativa completa con cada método de pitch-shift.
def exp_sintesis(base_audio):
    db = V.build_guitar_unit_db(base_audio, SR)
    times, f0, energy = V.track_dominant_per_frame(base_audio, SR)
    voices = {}
    for m in METHODS:
        v = V.concatenative_voice(base_audio, SR, times, f0, energy, db, method=m, gate=0.4)
        voices[m] = v
        sf.write(os.path.join(OUT, 'voz_%s.wav' % m), V.normalize_audio(v), SR)
    mixed = V.mix_over(base_audio, voices['pv_formant'], mix=0.55)
    sf.write(os.path.join(OUT, 'mezcla_pv_formant.wav'), mixed, SR)
    sf.write(os.path.join(OUT, 'guitarra_base.wav'), V.normalize_audio(base_audio), SR)

    # FIG6: formas de onda de las tres voces
    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for a, m in zip(ax, METHODS):
        a.plot(np.arange(len(voices[m])) / SR, voices[m], lw=0.5, color=METHOD_COLOR[m])
        a.set_title('Voz concatenativa — %s' % METHOD_LABEL[m], fontsize=10); a.set_ylabel('amp'); a.grid(alpha=0.3)
    ax[-1].set_xlabel('tiempo [s]')
    fig.suptitle('Síntesis concatenativa "Pedal Miku" con cada método de pitch-shift',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig06_sintesis_ondas.png'), dpi=130); plt.close(fig)

    # FIG7: espectrogramas de las tres voces
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, m in zip(ax, METHODS):
        plot_spectro(a, voices[m], METHOD_LABEL[m])
    fig.suptitle('Espectrogramas de la voz sintetizada por método', fontsize=12, fontweight='bold')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig07_sintesis_espectros.png'), dpi=130); plt.close(fig)

    results['sintesis'] = {m: {'dur_s': len(voices[m]) / SR,
                               'centroid': V.spectral_centroid(voices[m])} for m in METHODS}
    print('SINTESIS: voces generadas (%d unidades en la base)' % len(db))


# ============================================================================= main
def main():
    print('=== Generando resultados (semilla=%d) ===' % SEED)
    guitar_path = os.path.join(DATA, 'guitarra.mp3')
    if not os.path.exists(guitar_path):
        import descargar_datos
        descargar_datos.main()
    base_audio = V.load_audio(guitar_path, SR)
    # Recortar a 8 s para acotar el tiempo de cómputo del phase vocoder por grano.
    base_audio = V.normalize_audio(base_audio[:int(8.0 * SR)])
    print('Guitarra: %.2f s (%d muestras)' % (len(base_audio) / SR, len(base_audio)))

    fig_reproduccion()
    exp_tono_duracion()
    exp_formantes()

    db = V.build_guitar_unit_db(base_audio, SR)
    grain = db[len(db) // 2]['wave'] if db else flat_tone(200.0, 0.14)
    fig_grano_real(grain)
    exp_sintesis(base_audio)

    with open(os.path.join(HERE, 'resultados.json'), 'w', encoding='utf-8') as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print('OK -> resultados.json + figuras/ + outputs/')


if __name__ == '__main__':
    main()
