# Autotune con voz de Miku — Proyecto Semestral PDDI

Un **autotune** en DSP puro: toma **tu voz** cantada, **corrige la afinación** a las notas de una
escala, la lleva al **registro de Miku** y le **transfiere el timbre (formantes) de Hatsune Miku**,
conservando tu interpretación (melodía y ritmo). El motor reproduce el **phase vocoder** de Dolson
(1986) y una **extensión propia** (preservación de formantes por cepstrum). **Solo técnicas clásicas
del curso** (Fourier, STFT, enventanado, filtrado, cepstrum, autocorrelación/pYIN); **sin machine
learning, deep learning ni LLM**.

**Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1**
Francisco Alejandro Pinto Abraham — RUT 21.571.239-7

**Artículo base:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, *Computer Music Journal*, 10(4):14–27, 1986.

> **Cumplimiento de la pauta:** ver [`ENTREGABLES.md`](ENTREGABLES.md) — mapea cada entregable, etapa y
> criterio de la rúbrica a su archivo.

## Abrir en Colab (la aplicación principal)

👉 https://colab.research.google.com/github/k0ngan/mikustromp/blob/main/miku_autotune_colab.ipynb

**Entorno de ejecución → Ejecutar todo.** Trae la voz de Miku incluida (referencia de timbre), así que
suena a Miku **sin subir nada**. Funciona **en vivo con audios nuevos**: sube un `.wav`/`.mp3`, **graba
con el micrófono** (`USE_MIC = True`) o usa la voz de prueba.

## Idea en una línea

Afinar una voz a una escala **sin cambiar su duración** (phase vocoder) y **darle el timbre y el registro
de Miku** (transferencia de formantes por cepstrum) — un autotune completo en DSP clásico.

## Resultados principales

**Autotune** (voz de prueba con verdad de terreno, Do mayor):

| Métrica | Antes | Después |
|---|---|---|
| Error de afinación medio | **~42 cents** | **~5 cents** |
| Error de duración | 0 % | **0 %** |
| Timbre: LSD de la envolvente a Miku | ~5.3 dB | **~3.3 dB** |

**Motor (phase vocoder), validación del pitch-shift:**

| Método | Error duración | Error afinación | Formantes (LSD env.) |
|---|---|---|---|
| Remuestreo (baseline) | hasta ±50 % | < 1 cent | ~8.6 dB (se mueven) |
| Phase vocoder (Dolson) | **0 %** | < 8 cents | ~7.8 dB (se mueven) |
| **PV + formantes (extensión)** | **0 %** | < 8 cents | **~1.6 dB (se conservan)** |

## Archivos

| Archivo | Descripción |
|---|---|
| **`autotune.py`** | **Motor del autotune Miku:** snap a la escala, corrección de tono variable en el tiempo (bloques + overlap-add), registro automático de Miku, acondicionado de voz, transferencia de formantes. Reusa `vocoder.py`. |
| `vocoder.py` | Núcleo DSP: STFT/ISTFT, phase vocoder, pitch-shift (resample/PV/PV+formantes), cepstrum, métricas. Reutiliza funciones de `miku_pedal.ipynb` (marcadas `[Reutilizado]`). |
| `stomp.py` | Aplicación secundaria (Miku Stomp de guitarra): pYIN + segmentación de notas + síntesis. |
| `generar_resultados_autotune.py` | Experimentos del autotune → `figuras/at_*.png`, `outputs/at_*.wav`, `resultados_autotune.json`. Semilla 2026. |
| `generar_resultados.py` | Experimentos del motor (phase vocoder) → `figuras/fig*.png`, `outputs/*.wav`, `resultados.json`. |
| `construir_autotune.py` | Arma `miku_autotune_colab.ipynb` (embebe el núcleo DSP + la voz de Miku). |
| `construir_notebook.py` / `construir_stomp.py` | Arman `phase_vocoder_pddi.ipynb` y `miku_stomp_colab.ipynb`. |
| **`miku_autotune_colab.ipynb`** | **Cuaderno principal** (Colab autocontenido) del autotune Miku. |
| `phase_vocoder_pddi.ipynb` | Laboratorio reproducible del motor (phase vocoder). |
| `miku_stomp_colab.ipynb` | Cuaderno de la aplicación secundaria (Miku Stomp). |
| `generar_informe.py` / `informe_phase_vocoder_pddi.pdf` | Genera y contiene el **informe** (eje: autotune). |
| `generar_presentacion_pdf.py` / `presentacion/presentacion_autotune_miku_pddi.pdf` | **Presentación** (diapositivas 16:9). |
| `descargar_datos.py` | Descarga idempotente de `data/guitarra.mp3`. |
| `figuras/`, `outputs/`, `data/`, `voces/` | Figuras, audios, caché de audio y la voz de Miku (`voces/miku_voice.wav`, Vocaloid 4 CyberDiva, uso académico). |

## Autotune (Colab): perillas y uso

- **KEY / SCALE:** tonalidad y escala destino (`chromatic`, `major`, `minor`, `pentatonic`).
- **STRENGTH** (0–1): dureza del "snap" (1 = pega del todo).
- **RETUNE_SPEED** (0–1): 1 = enganche instantáneo ("T-Pain"); bajo = glissando suave.
- **MIKU_AMOUNT** (0–1): cuánto timbre de Miku se impone.
- **OCTAVE:** `"auto"` lleva la voz al registro de Miku (sube octavas); un entero fija la octava.
- **GATE:** compuerta de silencios. La voz de entrada se **acondiciona** sola (quita ruido/DC, saca el
  click de grabación y sube el nivel si venía floja).

## Reproducir desde cero

Requisitos (Python 3.12): `numpy scipy matplotlib soundfile librosa reportlab nbformat jupyter pillow`.

```bash
cd proyecto_phase_vocoder
python descargar_datos.py             # obtiene data/guitarra.mp3 (una vez)
python generar_resultados.py          # motor: figuras fig*.png + outputs/ + resultados.json
python generar_resultados_autotune.py # autotune: at_*.png + outputs/at_*.wav + resultados_autotune.json
python construir_autotune.py          # arma miku_autotune_colab.ipynb (principal)
python construir_notebook.py          # arma phase_vocoder_pddi.ipynb (laboratorio del motor)
jupyter nbconvert --to notebook --execute --inplace miku_autotune_colab.ipynb
python generar_informe.py             # informe PDF (autotune como eje)
python generar_presentacion_pdf.py    # diapositivas PDF
```

Reproducible: semilla fija (`SEED = 2026`). Las figuras y tablas del informe se generan desde el código.

## Código reutilizado vs. propio (declaración)

- **Reutilizado** del proyecto previo `miku_pedal.ipynb`: utilidades de señal, análisis de Fourier, base
  de granos y síntesis concatenativa (marcadas `[Reutilizado]` en `vocoder.py`).
- **Desarrollo propio:** el phase vocoder, la preservación de formantes por cepstrum, todas las métricas,
  los experimentos, el **motor de autotune** (`autotune.py`: snap a escala, corrección variable en el
  tiempo, registro de Miku, acondicionado de voz y transferencia de formantes) y `stomp.py`.
- Se usó asistencia de IA como apoyo de programación y redacción, revisada por el autor.

## Referencias

1. M. Dolson, *The Phase Vocoder: A Tutorial*, Computer Music Journal 10(4), 1986.
2. J. L. Flanagan, R. M. Golden, *Phase Vocoder*, Bell System Technical Journal 45, 1966.
3. J. Laroche, M. Dolson, *Improved phase vocoder time-scale modification of audio*, IEEE TSAP 7(3), 1999.
