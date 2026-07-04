# Entregables y cumplimiento de la pauta — Proyecto Semestral PDDI

**Curso:** Procesamiento Digital de Señales e Imágenes (INFB6063) — UTEM 2026-1
**Estudiante:** Francisco Alejandro Pinto Abraham — RUT 21.571.239-7
**Artículo:** M. Dolson, *"The Phase Vocoder: A Tutorial"*, Computer Music Journal 10(4):14–27, 1986.
**Repositorio:** https://github.com/k0ngan/mikustromp

## 1. Entregables (Sección 5 de la pauta)

| Entregable exigido | Archivo en el repo |
|---|---|
| Informe técnico en PDF | `informe_phase_vocoder_pddi.pdf` (eje: autotune Miku) |
| Notebook Jupyter ejecutable | `miku_autotune_colab.ipynb` (**aplicación principal**, Colab), `phase_vocoder_pddi.ipynb` (laboratorio del motor) y `miku_stomp_colab.ipynb` (aplicación secundaria) |
| Código fuente | `autotune.py`, `vocoder.py`, `stomp.py`, `generar_resultados.py`, `generar_resultados_autotune.py`, `generar_informe.py`, `construir_autotune.py`, `construir_notebook.py`, `construir_stomp.py`, `generar_presentacion_pdf.py`, `descargar_datos.py` |
| Datos o enlace | `data/guitarra.mp3` (+ URL `raw.githubusercontent.com/k0ngan/audio/main/guitarra.mp3`); voz `voces/miku_voice.wav` (Vocaloid 4 CyberDiva, uso académico) |
| Presentación oral | `presentacion/presentacion_phase_vocoder_stomp.pdf` (diapositivas) y `presentacion/phase_vocoder_presentacion.mp4` (video narrado) |

Reproducibilidad: el notebook se ejecuta de principio a fin sin intervención manual; todas las figuras
y tablas del informe se generan desde el código (`generar_resultados.py`, semilla fija `SEED=2026`).

## 2. Etapas (Sección 4)

| Etapa | Dónde se cumple |
|---|---|
| 1. Comprensión del artículo | Informe Sec. 1-2; notebook Sec. 1 (en palabras propias). |
| 2. Reproducción de la metodología | Phase vocoder en `vocoder.py`; informe Sec. 4 y Fig. 1; notebook Sec. 3. |
| 3. Modificación/extensión/aplicación | Extensión: preservación de formantes (Sec. 5). Aplicación principal: **Autotune con voz de Miku** (Sec. 8); secundaria: Miku Stomp (Sec. 9). |
| 4. Evaluación experimental | Informe Sec. 6-9: afinación antes/después y timbre del autotune (Fig. 8-11), tono/duración y formantes del motor (Fig. 2-4), afinación del stomp (Fig. 12-13). |
| 5. Discusión final (7 preguntas) | Informe Sec. 10 (responde las 7 explícitamente); notebook Sec. 9. |

## 3. Estructura del informe (Sección 6)

Introducción (1) · Resumen del artículo (2) · Relación con el curso (3) · Metodología original (4) ·
Extensión propuesta (5) · Diseño experimental (6) · Resultados (7) · Aplicación principal: Autotune
con voz de Miku (8) · Aplicación secundaria: Miku Stomp (9) · Discusión (10) · Conclusiones (11) ·
Código reutilizado (12) · Referencias.

## 4. Rúbrica (Sección 9)

| Criterio | Dónde |
|---|---|
| Comprensión del artículo (20%) | Informe Sec. 1-2, 4. |
| Relación con el curso (15%) | Informe Sec. 3 (tabla concepto→unidad). |
| Reproducción (20%) | `vocoder.py` + Fig. 1; resultados intermedios en el notebook. |
| Modificación/extensión (15%) | Formantes (Sec. 5) + aplicación principal Autotune Miku (Sec. 8) + Miku Stomp (Sec. 9). |
| Diseño experimental (10%) | Sec. 6: métricas justificadas (cents, % duración, LSD, F1). |
| Análisis y discusión (15%) | Sec. 9: casos de éxito/falla, causas, 7 preguntas. |
| Presentación/informe/reproducibilidad (5%) | Informe + slides + video + notebook reproducible. |

## 5. Restricciones y declaración (Sección 2 y 10)

- **Sin deep learning / IA generativa como eje.** Todo es DSP clásico: Fourier/DFT, STFT, enventanado,
  phase vocoder, cepstrum, overlap-add, muestreo, autocorrelación y **pYIN** (YIN probabilístico, no es
  red neuronal). RVC / so-vits / OpenUTAU se mencionan solo como trabajo futuro.
- **Código reutilizado vs propio:** se reutilizan utilidades, análisis de Fourier, base de granos y
  síntesis concatenativa del proyecto previo `miku_pedal.ipynb` (marcadas `[Reutilizado]` en
  `vocoder.py`). Son desarrollo propio: el phase vocoder, la preservación de formantes por cepstrum,
  las métricas, los experimentos, el **motor de autotune** (`autotune.py`: snap a escala, corrección
  variable en el tiempo y transferencia de formantes de Miku) y `stomp.py`. Se usó asistencia de IA como
  apoyo, revisada por el autor.

## 6. Pendientes administrativos (fuera del repo)

- Inscribir el grupo (máx. 2 estudiantes) en la tabla de la pauta.
- Obtener la **aprobación del artículo por el profesor** (Etapa 1) antes de la entrega final.
