# Phase Vocoder (Dolson 1986) + "Pedal Miku" — Proyecto Semestral PDDI

Reproducción del **phase vocoder** de Dolson y una **extensión propia** (preservación de
formantes), aplicadas a un sintetizador concatenativo de voz hecha con granos de guitarra.
**Solo técnicas clásicas del curso** (Fourier, STFT, enventanado, filtrado, cepstrum); sin
machine learning, deep learning ni LLM.

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1**
Francisco Alejandro Pinto Abraham — RUT 21.571.239-7

**Artículo:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, *Computer Music Journal*, 10(4):14–27, 1986.

> **Cumplimiento de la pauta:** ver [`ENTREGABLES.md`](ENTREGABLES.md) — mapea cada entregable, etapa y
> criterio de la rúbrica a su archivo.

## Idea en una línea

Cambiar el **tono** de un sonido **sin cambiar su duración** (phase vocoder) y, como extensión,
**sin mover los formantes** (timbre), comparándolo con el remuestreo ingenuo.

## Resultados principales

| Método | Error duración | Error afinación | Formantes (LSD env.) |
|---|---|---|---|
| Remuestreo (baseline) | hasta ±50 % | < 1 cent | ~8.6 dB (se mueven) |
| Phase vocoder (Dolson) | **0 %** | < 8 cents | ~7.8 dB (se mueven) |
| **PV + formantes (extensión)** | **0 %** | < 8 cents | **~1.6 dB (se conservan)** |

## Archivos

| Archivo | Descripción |
|---|---|
| `vocoder.py` | Núcleo DSP: STFT/ISTFT, phase vocoder, pitch-shift (resample/PV/PV+formantes), métricas. Reutiliza funciones de `miku_pedal.ipynb` (marcadas `[Reutilizado]`). |
| `descargar_datos.py` | Descarga idempotente de `guitarra.mp3` (respaldo local si no hay red). |
| `generar_resultados.py` | Corre los experimentos → `figuras/`, `outputs/*.wav`, `resultados.json`. Semilla fija. |
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
python descargar_datos.py        # obtiene guitarra.mp3 (una vez)
python generar_resultados.py     # figuras/ + outputs/ + resultados.json
python construir_notebook.py     # arma el notebook
jupyter nbconvert --to notebook --execute --inplace phase_vocoder_pddi.ipynb
python generar_informe.py        # informe PDF
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
