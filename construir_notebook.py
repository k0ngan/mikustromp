# -*- coding: utf-8 -*-
"""
construir_notebook.py -- Arma el notebook reproducible `phase_vocoder_pddi.ipynb`.

El notebook es un informe-laboratorio EJECUTABLE de punta a punta: importa `vocoder.py`,
muestra la reproducción del phase vocoder con audio, ejecuta los experimentos
(`generar_resultados.main()`) y despliega figuras + reproductores de audio + tablas.
"""

import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = []

cells.append(md(r"""# Phase Vocoder (Dolson 1986) + "Pedal Miku" — Proyecto Semestral PDDI

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM, 2026-1**
**Estudiante:** Francisco Alejandro Pinto Abraham · **RUT:** 21.571.239-7

**Artículo reproducido:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, *Computer Music Journal*, 10(4):14–27, 1986.

Este notebook se ejecuta de principio a fin sin intervención manual. Toda la lógica DSP vive en
`vocoder.py` (reutiliza funciones del proyecto previo `miku_pedal.ipynb`, declarado explícitamente);
los experimentos los corre `generar_resultados.py`.
"""))

cells.append(md(r"""## 1. El problema y el artículo (en mis palabras)

**Problema.** Queremos **cambiar el tono** de un sonido (subirlo/bajarlo en semitonos) **sin cambiar
su duración** y, ojalá, **sin deformar el timbre**. El camino ingenuo —releer la señal más rápido o
más lento (remuestreo)— sube el tono pero **acorta/alarga** el audio y **corre los formantes** (el
clásico efecto "ardilla").

**El aporte de Dolson (1986).** El *phase vocoder* separa, en cada banda de frecuencia de la STFT, la
**magnitud** (cuánta energía hay) de la **fase** (dónde va esa oscilación). Reconstruyendo con un
**salto de síntesis distinto** al de análisis se cambia la **duración**; la clave es **propagar la
fase** usando la **frecuencia instantánea** de cada banda (estimada por la diferencia de fase entre
tramas, corregida módulo 2π) para que las componentes sigan siendo coherentes. Un **pitch-shift** se
logra entonces como **time-stretch + remuestreo**.

**Relación con el curso.** Es DSP clásico: **STFT y enventanado Hann (U2-L1)**, **DFT/espectro y fase
(U1-L2/L3)**, **overlap-add (U2-L1/L2)**, **muestreo/interpolación (U1-L3)** y, para la extensión,
**filtrado en frecuencia como máscara (U2-L3)** y **cepstrum**. Sin deep learning / ML / LLM.

**Datos.** Una señal real de **guitarra** (`guitarra.mp3`, repositorio público de GitHub) y **señales
sintéticas** con f0 y formantes **conocidos** (verdad de terreno para medir el error).

**Limitaciones reportadas (y observadas).** El phase vocoder introduce *phasiness* (difuminado por
pérdida de coherencia de fase vertical) y **emborrona transitorios**; el pitch-shift básico **mueve
los formantes** igual que el remuestreo.
"""))

cells.append(md(r"""## 2. Qué reproduzco y qué propongo

- **Reproducción (Etapa 2):** el phase vocoder de Dolson — STFT → frecuencia instantánea por fase →
  *time-stretch* → *pitch-shift*.
- **Extensión propia (Etapa 3):** un **phase vocoder con preservación de formantes**: tras subir el
  tono, **re-impongo la envolvente espectral original** (estimada por **cepstrum**) con una máscara
  `H = env_orig / env_desplazada`. Así los armónicos se mueven pero los **formantes quedan fijos**.
- **Comparación (Etapa 4):** tres métodos de pitch-shift dentro del sintetizador concatenativo
  "Pedal Miku": **(A) remuestreo ingenuo** (baseline), **(B) phase vocoder**, **(C) PV + formantes**.
"""))

cells.append(code(r"""import os, json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, Image, display

import vocoder as V
print('vocoder importado | SR =', V.SR, 'Hz')

# Asegurar el audio de guitarra (descarga idempotente con respaldo local)
import descargar_datos
guitar_path = descargar_datos.main()
"""))

cells.append(md(r"""## 3. Reproducción del phase vocoder: cambiar la duración SIN cambiar el tono

Estiramos un tono de prueba a 1.5× su duración con el phase vocoder. La **forma de onda dura más**,
pero el **tono (frecuencia) se mantiene**: esa es la firma del método de Dolson.
"""))

cells.append(code(r"""x = V.normalize_audio(np.sin(2*np.pi*220.0*np.arange(int(0.6*V.SR))/V.SR).astype('float32'))
slow = V.time_stretch_pv(x, 1.5)   # 50% mas largo, mismo tono

def domf(s):
    f,m = V.compute_rfft(s); b=(f>=60)&(f<=1500); return f[b][np.argmax(m[b])]

print('Original : %.2f s, f0=%.0f Hz' % (len(x)/V.SR, domf(x)))
print('x1.5     : %.2f s, f0=%.0f Hz  (duracion +50%%, tono igual)' % (len(slow)/V.SR, domf(slow)))

fig, ax = plt.subplots(1,2, figsize=(12,3))
ax[0].plot(np.arange(len(x))/V.SR, x, lw=0.5); ax[0].set_title('Original 220 Hz'); ax[0].grid(alpha=.3)
ax[1].plot(np.arange(len(slow))/V.SR, slow, lw=0.5, color='tab:orange'); ax[1].set_title('Time-stretch x1.5'); ax[1].grid(alpha=.3)
for a in ax: a.set_xlim(0, len(slow)/V.SR); a.set_xlabel('s')
plt.tight_layout(); plt.show()

print('Escucha original vs estirado (mismo tono, mas largo):')
display(Audio(x, rate=V.SR)); display(Audio(slow, rate=V.SR))
"""))

cells.append(md(r"""## 4. Los tres métodos de pitch-shift

`vocoder.py` expone los tres como `PITCH_METHODS['resample' | 'pv' | 'pv_formant']`. Subimos una
vocal sintética **+5 semitonos** y comparamos: el **remuestreo** acorta y mueve formantes; el
**phase vocoder** conserva duración pero mueve formantes; **PV + formantes** conserva ambas cosas.
"""))

cells.append(code(r"""x = V.synth_vowel(150.0, 'a', dur=0.8, seed=2026)   # vocal /a/ con formantes conocidos
print('Vocal /a/ original: %.2f s' % (len(x)/V.SR)); display(Audio(x, rate=V.SR))
for m in ['resample','pv','pv_formant']:
    y = V.PITCH_METHODS[m](x, +5)
    print('%-12s -> %.2f s' % (m, len(y)/V.SR)); display(Audio(V.normalize_audio(y), rate=V.SR))
"""))

cells.append(md(r"""## 5. Experimentos (reproducibles, semilla fija)

Ejecutamos `generar_resultados.main()`: corre todos los experimentos, guarda las figuras en
`figuras/`, los audios en `outputs/` y las métricas en `resultados.json`.
"""))

cells.append(code(r"""import generar_resultados as G
G.main()
res = json.load(open('resultados.json', encoding='utf-8'))
print('Listo. Claves de resultados:', list(res.keys()))
"""))

cells.append(md(r"""### 5.1 Reproducción del phase vocoder"""))
cells.append(code(r"""display(Image('figuras/fig01_reproduccion_pv.png'))"""))

cells.append(md(r"""### 5.2 Exactitud de tono y de duración

El **remuestreo** cambia la duración hasta ±50 %; el **phase vocoder** la conserva (0 %). El error de
afinación es **sub-audible (< 5 cents)** en los tres métodos.
"""))
cells.append(code(r"""display(Image('figuras/fig02_tono_duracion.png'))
print('Resumen tono/duracion:')
for m,s in res['tono_duracion_resumen'].items():
    print('  %-12s cents(mean=%.1f, max=%.1f)  dur%%(mean=%.1f, max=%.1f)' %
          (m, s['cents_abs_mean'], s['cents_abs_max'], s['dur_abs_mean'], s['dur_abs_max']))
"""))

cells.append(md(r"""### 5.3 Preservación de formantes

Sólo **PV + formantes** mantiene el formante F1 cerca del original y minimiza la distancia
log-espectral (LSD) de la envolvente. Para tonos grandes (≥ +5–7 semitonos) la corrección se degrada
(caso de falla, discutido más abajo).
"""))
cells.append(code(r"""display(Image('figuras/fig03_formantes.png'))
display(Image('figuras/fig04_lsd.png'))
print('Resumen formantes:')
for m,s in res['formantes_resumen'].items():
    print('  %-12s |F1 shift| mean=%.0f Hz | LSD mean=%.1f dB | env-centroid ratio=%.2f' %
          (m, s['F1_abs_shift_mean'], s['lsd_mean_db'], s['env_centroid_ratio_mean']))
"""))

cells.append(md(r"""### 5.4 Caso real: un grano de guitarra desplazado"""))
cells.append(code(r"""display(Image('figuras/fig05_grano_real.png'))"""))

cells.append(md(r"""### 5.5 Integración: la voz "Pedal Miku" con cada método

La voz concatenativa hecha de granos de guitarra, sintetizada con cada método de pitch-shift.
"""))
cells.append(code(r"""display(Image('figuras/fig06_sintesis_ondas.png'))
display(Image('figuras/fig07_sintesis_espectros.png'))
for m in ['resample','pv','pv_formant']:
    print('Voz —', G.METHOD_LABEL[m])
    display(Audio('outputs/voz_%s.wav' % m))
print('Mezcla guitarra + voz (PV + formantes):')
display(Audio('outputs/mezcla_pv_formant.wav'))
"""))

cells.append(md(r"""### 5.6 Aplicación alternativa: Miku Stomp digital

Aplicación del método al flujo del pedal **Korg Miku Stomp** en DSP puro: **guitarra -> detectar la
nota (pYIN) -> afinar una voz a esa nota -> mezclar**. La voz sigue el contorno de tono de la guitarra
(glissando) con preservación de formantes. El cuaderno autocontenido para Colab es
`miku_stomp_colab.ipynb`. Afinación de la voz vs la guitarra:
"""))
cells.append(code(r"""display(Image('figuras/fig08_stomp_f0.png'))
display(Image('figuras/fig09_stomp_modos.png'))
st = res.get('stomp', {})
if st:
    print('Miku Stomp: %d notas, offset global = %d octavas, voz f0 = %.0f Hz' %
          (st['n_notas'], st['global_octave'], st['voice_f0']))
    for m, a in st['afinacion'].items():
        print('  %-12s error afinacion mediano = %s cents | dentro de 1 semitono = %s%%' %
              (m, round(a['cents_median']) if a['cents_median'] is not None else '-',
               round(a['within_semitone_pct']) if a['within_semitone_pct'] is not None else '-'))
for m in ['resample', 'pv_formant']:
    print('Stomp —', G.METHOD_LABEL[m]); display(Audio('outputs/stomp_%s.wav' % m))
print('Mezcla guitarra + voz (stomp):'); display(Audio('outputs/stomp_mezcla_final.wav'))
"""))

cells.append(md(r"""## 6. Discusión (Etapa 5)

- **Lo más difícil de comprender:** la **propagación de fase** por frecuencia instantánea y el
  *unwrapping* módulo 2π; sin esa corrección, el sonido se desfasa y aparece *phasiness*.
- **Lo más complejo de implementar:** la **preservación de formantes** — estimar la envolvente por
  cepstrum y elegir el lifter/rango de ganancia (±60 dB) para no amplificar valles ni ruido.
- **Conceptos del curso clave:** STFT/enventanado, fase de la DFT, overlap-add, muestreo/interpolación
  y filtrado en frecuencia como máscara.
- **¿La modificación mejoró el método?** Sí en **timbre**: PV + formantes baja la LSD de ~8 dB a
  ~1.6 dB y mantiene F1; **conserva** la ventaja de duración del phase vocoder. **No** mejora el tono
  (los tres ya afinan <5 cents) y **cuesta** más cómputo.
- **Cuándo funciona / cuándo falla:** funciona muy bien en desplazamientos **moderados** (−7…+3
  semitonos) sobre sonidos **cuasi-estacionarios**; **falla** en saltos grandes (≥ +5–7), donde la
  envolvente desplazada se solapa mal con la original y la corrección se degrada, y en **transitorios**
  (ataques de la guitarra), que el phase vocoder emborrona.
- **Con más tiempo:** *phase-locking* (Laroche–Dolson) para reducir phasiness, TD-PSOLA como
  comparador en el dominio del tiempo, y envolventes por LPC en vez de cepstrum.

## 7. Reproducibilidad y código reutilizado

Semilla fija (`SEED = 2026`). Las funciones de utilidad y la síntesis concatenativa provienen del
proyecto previo `miku_pedal.ipynb` (marcadas `[Reutilizado]` en `vocoder.py`); el phase vocoder, la
preservación de formantes, las métricas y todos los experimentos son **desarrollo propio** para este
proyecto.
"""))

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'authors': [{'name': 'Francisco Alejandro Pinto Abraham'}],
}

out = os.path.join(HERE, 'phase_vocoder_pddi.ipynb')
with open(out, 'w', encoding='utf-8') as fh:
    nbf.write(nb, fh)
print('Notebook escrito:', out, '(%d celdas)' % len(cells))
