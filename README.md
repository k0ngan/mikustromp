# Autotune con voz de Miku

Un **autotune** en DSP puro: toma **tu voz** cantada, **corrige la afinación** a las notas de una
escala, la lleva al **registro de Miku** y le **transfiere el timbre (formantes) de Hatsune Miku**,
conservando tu interpretación (melodía y ritmo). El motor reproduce el **phase vocoder** de Dolson
(1986) y una **extensión propia** (preservación de formantes por cepstrum). Todo con técnicas clásicas
de procesamiento de señales — **sin machine learning ni deep learning**.

**Proyecto Semestral — Procesamiento Digital de Señales e Imágenes (INFB6063), UTEM 2026-1**
Francisco Alejandro Pinto Abraham — RUT 21.571.239-7

**Artículo base:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, *Computer Music Journal*, 10(4):14–27, 1986.

## Qué hay en este repo

| Archivo | Qué es |
|---|---|
| [`miku_autotune_colab.ipynb`](miku_autotune_colab.ipynb) | Cuaderno **autocontenido** del autotune (corre en Colab o local, de principio a fin). |
| [`informe_phase_vocoder_pddi.pdf`](informe_phase_vocoder_pddi.pdf) | Informe técnico (método, experimentos y resultados). |

## Cómo usarlo

**Abrir en Google Colab:**
👉 https://colab.research.google.com/github/k0ngan/mikustromp/blob/main/miku_autotune_colab.ipynb

Luego: **Entorno de ejecución → Ejecutar todo.** El cuaderno trae **incluida** la voz de Miku (Vocaloid 4
CyberDiva) como referencia de timbre, así que suena a Miku **sin subir nada**.

**Funciona en vivo con audios nuevos:**
- **Subir** tu voz: deja `USE_MIC = False` y sube un `.wav`/`.mp3`.
- **Grabar en vivo** (Colab): pon `USE_MIC = True` y canta al ejecutar la celda de entrada.
- Sin nada: usa una **voz de prueba** desafinada para ver el efecto.

**Perillas del autotune:**
- `KEY` / `SCALE` — tonalidad y escala destino (`chromatic`, `major`, `minor`, `pentatonic`).
- `STRENGTH` (0–1) — dureza del "snap" (1 = pega del todo).
- `RETUNE_SPEED` (0–1) — 1 = enganche instantáneo ("T-Pain"); bajo = glissando suave.
- `MIKU_AMOUNT` (0–1) — cuánto timbre de Miku se impone.
- `OCTAVE` — `"auto"` lleva la voz al registro de Miku; un entero fija la octava.
- `GATE` — compuerta de silencios. La voz de entrada se **acondiciona** sola (quita ruido/DC, saca el
  click de grabación y sube el nivel si venía floja).

## Cómo funciona (en breve)

1. **Seguir el tono** de la voz cuadro a cuadro (pYIN / autocorrelación).
2. **Pegarlo a la escala** (`snap`): se pasa a número MIDI (`m = 69 + 12·log2(f/440)`) y se redondea a la
   nota de la escala más cercana.
3. **Corregir la afinación** variable en el tiempo con un **phase vocoder** que **preserva los formantes**,
   reconstruyendo por *overlap-add* → **la duración se conserva**.
4. **Registro y timbre de Miku**: se sube por octavas enteras al rango de Miku y se impone su **envolvente
   de formantes** (por **cepstrum**) como máscara espectral `H = env_Miku / env_voz`.

Solo DSP clásico del curso: FFT/STFT, enventanado, phase vocoder, cepstrum, *overlap-add*,
autocorrelación/pYIN (YIN probabilístico, **no** es una red neuronal).

## Límites (honestos)

Es **monofónico** (una voz a la vez) y funciona **por lotes** (graba/sube → procesa → reproduce), no en
*streaming* de latencia cero. Saltos de tono muy grandes o tomas muy silenciosas degradan el resultado.
Una versión en tiempo real (`sounddevice` / plugin) y la clonación de timbre con RVC/so-vits/OpenUTAU
quedan como **trabajo futuro** (fuera del eje DSP del ramo).
