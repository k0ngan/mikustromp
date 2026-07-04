# -*- coding: utf-8 -*-
"""
generar_resultados_autotune.py -- Experimentos y figuras del "Autotune con voz de Miku".

Genera, con semilla fija (SEED=2026) y de forma reproducible:
  - una VOZ de prueba desafinada con verdad de terreno conocida (secuencia de vocales fuera de tono),
  - la salida AFINADA (autotune) y la salida con TIMBRE de Miku (autotune + transferencia de formantes),
  - métricas antes/después: error de afinación en cents, error de duración (%), distancia log-espectral
    de la envolvente respecto a Miku, y centroides espectrales,
  - las figuras `figuras/at_*.png` y los audios `outputs/at_*.wav`,
  - el resumen `resultados_autotune.json`.

Reutiliza el núcleo DSP de `vocoder.py` y el motor de `autotune.py`. Todo es DSP clásico del curso.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import vocoder as V
import autotune as A

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figuras')
OUT = os.path.join(HERE, 'outputs')
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

SR = A.SR
NOTE_MS = 600.0                       # duración de cada nota de la melodía de prueba


# ------------------------------------------------------------------ utilidades locales
def load_miku_reference(dur=0.6):
    """Carga la voz real de Miku (voces/miku_voice.wav) y devuelve un grano estable como referencia de
    timbre. Si no existe, cae a una vocal sintética. Es E/S (no es materia del curso)."""
    path = os.path.join(HERE, 'voces', 'miku_voice.wav')
    if os.path.exists(path):
        y = V.load_audio(path, sr=SR)
        # tomar el tramo central de mayor energía como referencia estable
        L = int(dur * SR)
        if y.size > L:
            best_s, best_e = 0, -1.0
            for s in range(0, y.size - L, max(1, L // 4)):
                e = float(np.mean(y[s:s + L] ** 2))
                if e > best_e:
                    best_e, best_s = e, s
            y = y[best_s:best_s + L]
        return V.normalize_audio(y), 'miku_voice.wav (Vocaloid 4 CyberDiva)'
    return V.synth_vowel(f0=520.0, vowel='i', dur=dur, fs=SR, seed=SEED), 'vocal sintética /i/ (respaldo)'


def synth_offkey_voice(seed=SEED):
    """Qué hace: genera una VOZ de prueba (secuencia de vocales) cuyo tono está DESAFINADO respecto a la
    escala de Do mayor, con verdad de terreno conocida (nota objetivo y f0 real de cada segmento).
    Curso: suma de armónicos con envolvente de formantes (señal cuasi-vocálica); muestreo.
    Devuelve (x, verdad, limites) con verdad=[(midi_objetivo, f0_real)], limites=[(ini, fin) en muestras].
    """
    rng = np.random.default_rng(seed)
    midi_targets = [60, 62, 64, 65, 67, 69, 67, 64]           # C D E F G A G E (Do mayor)
    vowels = ['a', 'e', 'i', 'o', 'a', 'e', 'i', 'o']
    L = int(NOTE_MS / 1000.0 * SR)
    parts, truth, limits = [], [], []
    pos = 0
    for m, vw in zip(midi_targets, vowels):
        detune = float(rng.uniform(0.30, 0.5)) * (1 if rng.random() > 0.5 else -1)   # ±30..50 cents
        f_real = float(A.midi_to_freq(m + detune))
        seg = V.synth_vowel(f0=f_real, vowel=vw, dur=NOTE_MS / 1000.0, fs=SR,
                            seed=int(rng.integers(1, 10 ** 6)))
        seg = seg[:L] if seg.size >= L else np.pad(seg, (0, L - seg.size))
        parts.append(seg)
        truth.append((m, f_real))
        limits.append((pos, pos + L))
        pos += L
    return V.normalize_audio(np.concatenate(parts)), truth, limits


def measure_note_f0(y, ini, fin):
    """f0 medida en el tercio central de un segmento (evita los bordes/transitorios)."""
    a = ini + (fin - ini) // 3
    b = fin - (fin - ini) // 3
    return V.estimate_f0_autocorr(y[a:b], SR, 60.0, 1600.0)


def cents_to_note_octave_invariant(f_est, f_target):
    """Error de afinación en cents a la nota objetivo, INVARIANTE A LA OCTAVA (la salida se transpone al
    registro de Miku, así que se compara la clase de altura, no la octava). Rango: [-50, 50] cents."""
    if f_est <= 0 or f_target <= 0:
        return float('nan')
    semis = 12.0 * np.log2(f_est / f_target)
    return float((semis - round(semis)) * 100.0)          # desviación a la nota más cercana, en cents


def spectro_db(x, n_fft=1024, hop=256):
    X = V.stft(x, n_fft=n_fft, hop=hop)
    S = 20.0 * np.log10(np.abs(X).T + 1e-6)
    ff = np.fft.rfftfreq(n_fft, 1.0 / SR)
    tt = np.arange(X.shape[0]) * hop / SR
    return tt, ff, S


# ------------------------------------------------------------------ experimento principal
def main():
    np.random.seed(SEED)
    print('== Autotune Miku: generando resultados (semilla %d) ==' % SEED)

    miku_ref, miku_src = load_miku_reference()
    x, truth, limits = synth_offkey_voice()
    print('Voz de prueba: %d notas, %.2f s. Referencia de timbre: %s' %
          (len(truth), x.size / SR, miku_src))

    # pipeline completo: autotune (Do mayor) + timbre de Miku + registro de Miku (octava automática)
    res = A.mikutune(x, SR, key='C', scale='major', strength=1.0, retune_speed=1.0,
                     octave='auto', miku_amount=0.9, method='pv_formant', ref_miku=miku_ref)
    auto, miku, info = res['auto'], res['miku'], res['info']
    print('Registro de Miku: octava K=%+d aplicada.' % res['params']['octave_K'])

    # --- Métrica 1: afinación (cents) antes vs después, por nota (invariante a la octava) ---
    afin = []
    for (m, f_real), (ini, fin) in zip(truth, limits):
        f_target = float(A.midi_to_freq(m))
        f_before = measure_note_f0(x, ini, fin)
        f_after = measure_note_f0(auto, ini, fin)
        afin.append({
            'nota_midi': int(m),
            'f_objetivo': round(f_target, 2),
            'f_antes': round(float(f_before), 2),
            'cents_antes': round(cents_to_note_octave_invariant(f_before, f_target), 1),
            'f_despues': round(float(f_after), 2),
            'cents_despues': round(cents_to_note_octave_invariant(f_after, f_target), 1),
        })
    mae_before = float(np.mean([abs(a['cents_antes']) for a in afin]))
    mae_after = float(np.mean([abs(a['cents_despues']) for a in afin]))
    print('Afinación: |cents| medio  antes=%.1f  ->  después=%.1f' % (mae_before, mae_after))

    # --- Métrica 2: duración (debe conservarse) ---
    dur_err_auto = float(V.duration_error_pct(auto.size, x.size))
    dur_err_miku = float(V.duration_error_pct(miku.size, x.size))

    # --- Métrica 3: timbre -> distancia log-espectral de la envolvente respecto a Miku ---
    lsd_before = float(V.log_spectral_distance(miku_ref, x))
    lsd_auto = float(V.log_spectral_distance(miku_ref, auto))
    lsd_after = float(V.log_spectral_distance(miku_ref, miku))
    print('Timbre (LSD env. a Miku, dB):  entrada=%.2f  autotune=%.2f  +Miku=%.2f'
          % (lsd_before, lsd_auto, lsd_after))

    # --- Métrica 4: centroides ---
    cent = {
        'entrada': round(V.spectral_centroid(x), 1),
        'miku_ref': round(V.spectral_centroid(miku_ref), 1),
        'salida_miku': round(V.spectral_centroid(miku), 1),
        'env_entrada': round(V.envelope_centroid(x), 1),
        'env_miku_ref': round(V.envelope_centroid(miku_ref), 1),
        'env_salida_miku': round(V.envelope_centroid(miku), 1),
    }

    # ---------------- audios ----------------
    import soundfile as sf
    sf.write(os.path.join(OUT, 'at_entrada.wav'), x, SR)
    sf.write(os.path.join(OUT, 'at_autotune.wav'), auto, SR)
    sf.write(os.path.join(OUT, 'at_miku.wav'), miku, SR)
    sf.write(os.path.join(OUT, 'at_miku_ref.wav'), miku_ref, SR)

    # ---------------- figuras ----------------
    # Fig at_01: contorno de tono (medido, objetivo pegado a la escala, corregido)
    t = info['times']
    plt.figure(figsize=(12, 4))
    plt.plot(t, info['orig_midi'], '.', ms=5, color='tab:gray', label='tono medido (voz)')
    plt.plot(t, info['snapped_midi'], '_', ms=10, color='tab:red', label='nota de la escala (snap)')
    corrected = info['orig_midi'] + info['corr']
    plt.plot(t, np.where(info['voiced'], corrected, np.nan), '.', ms=4,
             color='tab:blue', label='tono corregido (autotune)')
    # líneas de las notas de Do mayor
    for m in range(58, 74):
        if ((m - 0) % 12) in set(A.SCALES['major']):
            plt.axhline(m, color='0.85', lw=0.6, zorder=0)
    plt.ylim(57, 73); plt.xlabel('tiempo [s]'); plt.ylabel('nota (MIDI)')
    plt.title('Autotune: el tono medido se pega a las notas de Do mayor'); plt.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'at_01_contorno.png'), dpi=130); plt.close()

    # Fig at_02: espectrograma entrada vs salida Miku
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, sig, ttl in [(ax[0], x, 'Entrada (voz desafinada)'),
                        (ax[1], miku, 'Salida (afinada + timbre Miku)')]:
        tt, ff, S = spectro_db(sig)
        mm = ff <= 4000
        a.pcolormesh(tt, ff[mm], S[mm], shading='auto', cmap='magma',
                     vmin=S.max() - 70, vmax=S.max())
        a.set_title(ttl); a.set_xlabel('s'); a.set_ylabel('Hz')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'at_02_espectrograma.png'), dpi=130); plt.close()

    # Fig at_03: envolventes de formantes (entrada vs Miku vs salida)
    def env_db(sig):
        f, mag = V.compute_rfft(sig, SR)
        e = V.cepstral_envelope(mag, 30)
        e = e / (np.mean(e) + 1e-9)
        return f, 20.0 * np.log10(e + 1e-9)
    plt.figure(figsize=(12, 4))
    for sig, c, lab in [(x, 'tab:gray', 'entrada (tu voz)'),
                        (miku_ref, 'tab:purple', 'Miku (referencia)'),
                        (miku, 'tab:green', 'salida (autotune + Miku)')]:
        f, e = env_db(sig)
        plt.plot(f, e, color=c, lw=1.6, label=lab)
    plt.xlim(0, 4000); plt.xlabel('Hz'); plt.ylabel('envolvente [dB]')
    plt.title('Transferencia de formantes: la salida adopta la envolvente de Miku')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'at_03_formantes.png'), dpi=130); plt.close()

    # Fig at_04: afinación en cents por nota, antes vs después
    idx = np.arange(len(afin))
    plt.figure(figsize=(12, 4))
    plt.bar(idx - 0.2, [a['cents_antes'] for a in afin], width=0.4, color='tab:gray', label='antes')
    plt.bar(idx + 0.2, [a['cents_despues'] for a in afin], width=0.4, color='tab:blue', label='después')
    plt.axhline(0, color='k', lw=0.8)
    plt.axhline(50, color='tab:red', lw=0.6, ls='--'); plt.axhline(-50, color='tab:red', lw=0.6, ls='--')
    plt.xticks(idx, ['n%d' % (i + 1) for i in idx])
    plt.ylabel('error de afinación [cents]')
    plt.title('Error de afinación por nota: antes vs después del autotune (0 = afinado)')
    plt.legend(); plt.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(FIG, 'at_04_afinacion.png'), dpi=130); plt.close()

    # ---------------- json ----------------
    resultados = {
        'seed': SEED,
        'sr': SR,
        'referencia_timbre': miku_src,
        'afinacion_por_nota': afin,
        'cents_abs_medio': {'antes': round(mae_before, 1), 'despues': round(mae_after, 1)},
        'duracion_error_pct': {'autotune': round(dur_err_auto, 3), 'miku': round(dur_err_miku, 3)},
        'lsd_envolvente_a_miku_dB': {'entrada': round(lsd_before, 2),
                                     'autotune': round(lsd_auto, 2),
                                     'miku': round(lsd_after, 2)},
        'centroides_hz': cent,
        'parametros': res['params'],
        'figuras': ['at_01_contorno.png', 'at_02_espectrograma.png',
                    'at_03_formantes.png', 'at_04_afinacion.png'],
        'audios': ['at_entrada.wav', 'at_autotune.wav', 'at_miku.wav', 'at_miku_ref.wav'],
    }
    with open(os.path.join(HERE, 'resultados_autotune.json'), 'w', encoding='utf-8') as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2)
    print('Escrito resultados_autotune.json + %d figuras + 4 audios.' % len(resultados['figuras']))
    return resultados


if __name__ == '__main__':
    main()
