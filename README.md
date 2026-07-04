# Autotune con voz de Miku — Proyecto Semestral PDDI

Un **autotune** que toma **tu voz** cantada, **corrige la afinación** a las notas de una escala
(el "snap") y le **transfiere el timbre (formantes) de Hatsune Miku** — conservando tu interpretación
(melodía y ritmo). El motor reproduce el **phase vocoder** de Dolson (1986) y una **extensión propia**
(preservación de formantes por cepstrum). **Solo técnicas clásicas del curso** (Fourier, STFT,
enventanado, filtrado, cepstrum, autocorrelación/pYIN); sin machine learning, deep learning ni LLM.

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1**
Francisco Alejandro Pinto Abraham — RUT 21.571.239-7

**Artículo:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, *Computer Music Journal*, 10(4):14–27, 1986.

> **Cumplimiento de la pauta:** ver [`ENTREGABLES.md`](ENTREGABLES.md) — mapea cada entregable, etapa y
> criterio de la rúbrica a su archivo.

## Idea en una línea

Afinar una voz a una escala **sin cambiar su duración** (phase vocoder) y **darle el timbre de Miku**
(transferencia de formantes por cepstrum) — un autotune completo en DSP puro.

## Resultados principales

**Autotune** (voz de prueba con verdad de terreno, Do mayor):

| Métrica | Antes | Después |
|---|---|---|
| Error de afinación medio | **~42 cents** | **~5 cents** |
| Error de duración | 0 % | **0 %** |
| Timbre: LSD de la envolvente a Miku | ~5.3 dB | **~4.2 dB** |

**Motor (phase vocoder), validación del pitch-shift:**

| Método | Error duración | Error afinación | Formantes (LSD env.) |
|---|---|---|---|
| Remuestreo (baseline) | hasta ±50 % | < 1 cent | ~8.6 dB (se mueven) |
| Phase vocoder (Dolson) | **0 %** | < 8 cents | ~7.8 dB (se mueven) |
| **PV + formantes (extensión)** | **0 %** | < 8 cents | **~1.6 dB (se conservan)** |

## Archivos

| Archivo | Descripción |
|---|---|
| `vocoder.py` | Núcleo DSP: STFT/ISTFT, phase vocoder, pitch-shift (resample/PV/PV+formantes), métricas. Reutiliza funciones de `miku_pedal.ipynb` (marcadas `[Reutilizado]`). |
| **`autotune.py`** | **Motor del autotune Miku:** snap a la escala, corrección de tono variable en el tiempo (por bloques + overlap-add) y transferencia de formantes de Miku. Reusa `vocoder.py`. |
| **`generar_resultados_autotune.py`** | Experimentos del autotune → figuras `at_*.png`, audios `outputs/at_*.wav`, `resultados_autotune.json`. Semilla 2026. |
| **`construir_autotune.py`** | Arma `miku_autotune_colab.ipynb` (embebe el núcleo DSP + la voz de Miku). |
| **`miku_autotune_colab.ipynb`** | **Cuaderno Colab autocontenido** del autotune Miku (sube/graba tu voz → afina + timbre Miku). |
| `descargar_datos.py` | Descarga idempotente de `guitarra.mp3` (respaldo local si no hay red). |
| `generar_resultados.py` | Corre los experimentos del phase vocoder → `figuras/`, `outputs/*.wav`, `resultados.json`. Semilla fija. |
| `construir_notebook.py` | Arma `phase_vocoder_pddi.ipynb` con `nbformat`. |
| `phase_vocoder_pddi.ipynb` | **Notebook reproducible** (informe-laboratorio ejecutable de punta a punta). |
| `generar_informe.py` | Genera el informe PDF (estructura tipo IEEE) con `reportlab`. |
| `informe_phase_vocoder_pddi.pdf` | **Informe formal**. |
| `presentacion/build_presentation.py` | Arma el video narrado (slides + TTS + ffmpeg). |
| `generar_presentacion_pdf.py` | Arma las **diapositivas PDF (16:9, beamer-like)** de la presentación oral. |
| `presentacion/presentacion_phase_vocoder_stomp.pdf` | **Presentación oral** (12 diapositivas: phase vocoder + Miku Stomp). |
| `stomp.py` | **Modo Stomp:** pYIN + segmentación de notas + síntesis tipo pedal (reusa `vocoder.py`). |
| `construir_stomp.py` | Arma `miku_stomp_colab.ipynb` (embebe el núcleo DSP). |
| `miku_stomp_colab.ipynb` | **Cuaderno Colab autocontenido** del "Miku Stomp digital". |
| `figuras/`, `outputs/`, `data/` | Figuras PNG, audios `.wav`, y caché del audio. |

## Autotune con voz de Miku (Colab): afina tu voz y ponle timbre de Miku

`miku_autotune_colab.ipynb` es la **aplicación principal**. Flujo en DSP puro:
**seguir el tono (pYIN) → pegarlo a la escala (snap) → corregir con phase vocoder preservando
formantes → imponer la envolvente de Miku (cepstrum)**. Conserva tu melodía y ritmo; solo cambia
afinación y timbre. **Funciona en vivo con audios nuevos:** sube un `.wav`/`.mp3`, **graba con el
micrófono** en Colab (`USE_MIC = True`), o usa una voz de prueba desafinada.

- **Perillas:** `KEY`/`SCALE` (escala destino: `chromatic`/`major`/`minor`/`pentatonic`), `STRENGTH`
  (0–1, dureza del snap), `RETUNE_SPEED` (0–1, 1 = enganche instantáneo "T-Pain"), `MIKU_AMOUNT`
  (0–1, cuánto timbre de Miku), `OCTAVE`.
- **Voz de Miku incluida** (Vocaloid 4 CyberDiva, uso académico) embebida como referencia de timbre.
- Autocontenido (embebe el núcleo DSP) y corre en **Colab** y **local** de principio a fin.
- **Límite honesto:** es monofónico y funciona **por lotes** (graba/sube → procesa → reproduce), no en
  streaming de latencia cero. Una versión en tiempo real (`sounddevice`/plugin) es trabajo futuro.

```bash
python generar_resultados_autotune.py   # figuras at_*.png + outputs/at_*.wav + resultados_autotune.json
python construir_autotune.py            # arma miku_autotune_colab.ipynb
jupyter nbconvert --to notebook --execute --inplace miku_autotune_colab.ipynb   # prueba local
```

## Modo Stomp (Colab): replicar el pedal Korg Miku Stomp

`miku_stomp_colab.ipynb` reproduce el flujo del pedal en **DSP puro**:
**guitarra → detectar la nota (pYIN, monofónico) → afinar una voz a esa nota → mezclar**.
Usa nuestro pitch-shift **con preservación de formantes**, así la voz mantiene su timbre al seguir la
melodía (el pedal original corre los formantes = efecto "ardilla"). Tres modos comparables:
`resample` (robótico, como el pedal), `pv`, y `pv_formant` (limpio, **mejor que el pedal**).

- **Voz:** sube un `.wav` real de Miku (un "aaah"); por defecto usa una **vocal sintética** con
  formantes reales (corre sin subir nada y sin temas de licencia).
- **pYIN** es un algoritmo clásico (YIN probabilístico), **no** una red neuronal. Sin ML/deep como eje;
  RVC / Basic Pitch / OpenUTAU se mencionan solo como **trabajo futuro**.
- El cuaderno es **autocontenido** (embebe el núcleo DSP) y corre tanto en **Colab** (subir/descargar)
  como **local** (con respaldos), de principio a fin.

```bash
python construir_stomp.py        # arma miku_stomp_colab.ipynb
jupyter nbconvert --to notebook --execute --inplace miku_stomp_colab.ipynb   # prueba local
```

## Reproducir desde cero

Requisitos (ya instalados en esta máquina): Python 3.12 con
`numpy scipy matplotlib soundfile reportlab nbformat jupyter pillow` (+ `edge-tts` y `ffmpeg`
para el video).

```bash
cd proyecto_phase_vocoder
python descargar_datos.py             # obtiene guitarra.mp3 (una vez)
python generar_resultados.py          # motor phase vocoder: figuras/ + outputs/ + resultados.json
python generar_resultados_autotune.py # autotune: at_*.png + outputs/at_*.wav + resultados_autotune.json
python construir_autotune.py          # arma miku_autotune_colab.ipynb (aplicación principal)
python construir_notebook.py          # arma phase_vocoder_pddi.ipynb (laboratorio del motor)
jupyter nbconvert --to notebook --execute --inplace miku_autotune_colab.ipynb
python generar_informe.py             # informe PDF (autotune como eje)
python presentacion/build_presentation.py   # video narrado (opcional)
```

Reproducible: semilla fija (`SEED = 2026`). Las figuras y tablas del informe se reproducen desde
el código.

## Código reutilizado

Las utilidades de señal, el análisis de Fourier, la base de granos y la síntesis concatenativa
provienen del proyecto previo `miku_pedal.ipynb` (declarado y marcado `[Reutilizado]` en
`vocoder.py`). Son **desarrollo propio**: el phase vocoder, la preservación de formantes por
cepstrum, todas las métricas y los experimentos. Se usó asistencia de IA como apoyo, revisada por
el autor.

## Referencias

1. M. Dolson, *The Phase Vocoder: A Tutorial*, Computer Music Journal 10(4), 1986.
2. J. L. Flanagan, R. M. Golden, *Phase Vocoder*, Bell System Technical Journal 45, 1966.
3. J. Laroche, M. Dolson, *Improved phase vocoder time-scale modification of audio*, IEEE TSAP 7(3), 1999.
