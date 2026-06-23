# -*- coding: utf-8 -*-
"""
generar_informe.py -- Informe formal en PDF (estructura tipo IEEE) a partir de las figuras de
figuras/ y las métricas de resultados.json, con reportlab.

Salida: informe_phase_vocoder_pddi.pdf
"""
import os
import json
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras")
OUT = os.path.join(HERE, "informe_phase_vocoder_pddi.pdf")

with open(os.path.join(HERE, "resultados.json"), encoding="utf-8") as f:
    R = json.load(f)
TD = R["tono_duracion_resumen"]
FO = R["formantes_resumen"]
RE = R["reproduccion"]

PAGE_W = A4[0]
MARGIN = 2.2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=13.5, spaceBefore=12,
                    spaceAfter=6, textColor=colors.HexColor("#0b3d91"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#264653"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=6)
CAP = ParagraphStyle("Cap", parent=styles["BodyText"], fontSize=8.5, leading=11,
                     alignment=TA_CENTER, textColor=colors.HexColor("#555"), spaceAfter=10)
TITLE = ParagraphStyle("Title", parent=styles["Title"], fontSize=17, leading=21,
                       textColor=colors.HexColor("#0b3d91"))
SUB = ParagraphStyle("Sub", parent=styles["BodyText"], fontSize=10.5, alignment=TA_CENTER, spaceAfter=2)
ABS = ParagraphStyle("Abs", parent=BODY, fontSize=9.5, leftIndent=14, rightIndent=14,
                     textColor=colors.HexColor("#222"))

story = []


def fig(nombre, ancho=CONTENT_W, caption=""):
    ruta = os.path.join(FIG, nombre)
    iw, ih = PILImage.open(ruta).size
    h = ancho * ih / iw
    story.append(Image(ruta, width=ancho, height=h))
    story.append(Paragraph(caption, CAP) if caption else Spacer(1, 8))


def p(t): story.append(Paragraph(t, BODY))
def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def sp(h=6): story.append(Spacer(1, h))


def tabla(data, col_w=None):
    t = Table(data, colWidths=col_w, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2fb")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t); sp(10)


# ===================================================================== Portada
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph("Cambio de tono sin alterar la duración: reproducción del "
                       "<i>phase vocoder</i> de Dolson y una extensión con preservación de formantes", TITLE))
sp(10)
story.append(Paragraph("Proyecto Semestral — Procesamiento Digital de Señales e Imágenes (INFB6063)", SUB))
story.append(Paragraph("Universidad Tecnológica Metropolitana (UTEM) — 2026-1", SUB))
sp(8)
story.append(Paragraph("Francisco Alejandro Pinto Abraham — RUT 21.571.239-7", SUB))
sp(14)
story.append(Paragraph("<b>Artículo reproducido:</b> M. Dolson, “The Phase Vocoder: A Tutorial”, "
                       "<i>Computer Music Journal</i>, vol. 10, n.º 4, pp. 14–27, 1986.", BODY))
sp(8)
story.append(Paragraph("<b>Resumen —</b> Se reproduce el <i>phase vocoder</i> de Dolson (1986), que "
    "modifica la duración de un sonido sin alterar su tono propagando la fase de la STFT mediante la "
    "frecuencia instantánea, y a partir de él un desplazamiento de tono (<i>pitch-shift</i>) como "
    "<i>time-stretch</i> + remuestreo. Como extensión propia se añade la <b>preservación de "
    "formantes</b> reimponiendo, vía cepstrum, la envolvente espectral original. Los tres métodos "
    "—remuestreo ingenuo (baseline), phase vocoder y phase vocoder con formantes— se integran en un "
    "sintetizador concatenativo de voz hecha con granos de guitarra (“Pedal Miku”) y se evalúan con "
    "señales sintéticas de verdad de terreno conocida y con audio real. El remuestreo deforma la "
    "duración hasta ±50 %%; el phase vocoder la conserva (0 %%) con error de afinación &lt; 5 cents; y "
    "la extensión reduce la distorsión de la envolvente de ~%.1f dB a ~%.1f dB manteniendo el "
    "formante F1. Sólo se usan técnicas clásicas del curso (Fourier, STFT, enventanado, filtrado, "
    "cepstrum); sin aprendizaje profundo." % (FO["resample"]["lsd_mean_db"], FO["pv_formant"]["lsd_mean_db"]), ABS))
sp(8)
story.append(Paragraph("<b>Palabras clave —</b> phase vocoder, STFT, frecuencia instantánea, "
                       "pitch-shifting, formantes, cepstrum, síntesis concatenativa.", BODY))
story.append(PageBreak())

# ===================================================================== 1. Introducción
h1("1. Introducción y descripción del problema")
p("Modificar el <b>tono</b> de un sonido sin cambiar su <b>duración</b> es una operación básica en "
  "música y voz (afinación, armonías, transposición). La vía ingenua —releer la señal a otra tasa "
  "(remuestreo)— sube o baja el tono pero <b>acorta o alarga</b> el audio y <b>corre los formantes</b>, "
  "produciendo el característico efecto “ardilla”. El reto es desacoplar tono, duración y timbre.")
p("Este trabajo reproduce el <i>phase vocoder</i> propuesto por Dolson [1], implementa con él un "
  "desplazamiento de tono que conserva la duración, y propone una <b>extensión</b> que además conserva "
  "los formantes. Los métodos se prueban dentro de un sintetizador concatenativo que arma una "
  "“voz” a partir de granos de una guitarra, reutilizado de un proyecto previo del curso.")

h1("2. Resumen explicado del artículo")
p("El phase vocoder analiza la señal con la <b>STFT</b>: en cada ventana se obtiene, por banda de "
  "frecuencia, una <b>magnitud</b> y una <b>fase</b>. La idea de Dolson es que la fase guarda la "
  "información de <b>frecuencia instantánea</b>: comparando la fase de una banda entre dos tramas "
  "consecutivas y restando el avance de fase esperado (corrigiendo módulo 2pi, <i>unwrapping</i>), se "
  "obtiene la frecuencia real de esa componente. Para cambiar la duración se reconstruye con un "
  "<b>salto de síntesis</b> distinto al de análisis, <b>acumulando</b> la fase con esos incrementos "
  "para mantener la coherencia. La síntesis es un <b>overlap-add</b> de las tramas inversas "
  "enventanadas. Un <b>pitch-shift</b> se obtiene combinando un <i>time-stretch</i> por un factor "
  "<i>r</i> y un remuestreo por <i>1/r</i>: el resultado sube el tono pero mantiene el largo.")
p("Dolson reporta como limitaciones la <i>phasiness</i> (sonido difuso por pérdida de coherencia "
  "entre bandas) y el <b>emborronado de transitorios</b>; además, el pitch-shift básico desplaza los "
  "formantes, porque el remuestreo escala todo el espectro, incluida su envolvente.")

h1("3. Relación con los contenidos del curso")
tabla([
    ["Etapa del método", "Concepto del curso"],
    ["Análisis STFT y enventanado Hann", "Unit 2 – L1 (STFT, ventanas)"],
    ["Espectro, magnitud y fase de la DFT", "Unit 1 – L2/L3 (DFT/FFT)"],
    ["Frecuencia instantánea (dif. de fase, unwrapping)", "Unit 1 – L2 (fase y espectro)"],
    ["Síntesis por superposición (overlap-add)", "Unit 2 – L1/L2 (enventanado, superposición)"],
    ["Pitch-shift = stretch + remuestreo", "Unit 1 – L3 (muestreo/interpolación)"],
    ["Preservación de formantes (máscara H[k])", "Unit 2 – L3 (filtro como máscara) + cepstrum"],
], col_w=[CONTENT_W * 0.55, CONTENT_W * 0.45])

h1("4. Metodología original (reproducción)")
p("Se implementó el phase vocoder en NumPy (módulo <font face='Courier'>vocoder.py</font>): "
  "<font face='Courier'>stft/istft</font> con ventana Hann y normalización de overlap-add; "
  "<font face='Courier'>phase_vocoder</font>, que propaga la fase por frecuencia instantánea e "
  "interpola la magnitud entre tramas; <font face='Courier'>time_stretch_pv</font> y "
  "<font face='Courier'>pitch_shift_pv</font>. La Fig. 1 verifica la reproducción: al estirar un tono "
  "de %.0f Hz a 1.5× su duración, el largo aumenta 50 %% mientras la frecuencia se mantiene "
  "(%.0f Hz a %.0f Hz)." % (RE["f0_original"], RE["f0_original"], RE["f0_stretch_x1.5"]))
fig("fig01_reproduccion_pv.png", caption="Fig. 1. Reproducción del phase vocoder: la duración cambia "
    "(time-stretch ×1.5) y el tono se conserva (espectrograma con las mismas frecuencias).")

h1("5. Extensión propuesta: phase vocoder con preservación de formantes")
p("El pitch-shift básico mueve los formantes junto con los armónicos. Como <b>extensión</b> propia se "
  "estima la <b>envolvente espectral</b> por <b>cepstrum</b> (suavizado del log-espectro con un lifter "
  "de baja cuefrencia) antes y después del desplazamiento, y se reimpone la envolvente original con una "
  "máscara <font face='Courier'>H[k] = env_orig / env_desplazada</font> (limitada a ±60 dB). Así los "
  "armónicos suben de tono pero los formantes —la identidad de la vocal— quedan fijos. Es filtrado en "
  "frecuencia (Unit 2 – L3) guiado por cepstrum, sin ningún componente de aprendizaje.")

h1("6. Diseño experimental")
p("Se comparan tres métodos —<b>remuestreo</b> (baseline), <b>phase vocoder</b> (PV) y "
  "<b>PV + formantes</b>— en desplazamientos de -7 a +12 semitonos. Para medir <b>tono</b> y "
  "<b>duración</b> se usan tonos de espectro plano (el fundamental domina y f0 es medible por pico "
  "espectral); para medir <b>formantes</b> se usan vocales sintéticas con F1/F2/F3 conocidos. Métricas: "
  "error de afinación en <b>cents</b>, error de <b>duración</b> en %%, posición del formante <b>F1</b>, "
  "<b>distancia log-espectral (LSD)</b> de la envolvente (invariante a ganancia) y razón de centroide "
  "de la envolvente. Semilla fija (2026); todo reproducible desde el código.")

h1("7. Resultados")
h2("7.1 Tono y duración")
tabla([
    ["Método", "|cents| medio", "|cents| máx", "|dur %| medio", "|dur %| máx"],
    ["Remuestreo", "%.2f" % TD["resample"]["cents_abs_mean"], "%.2f" % TD["resample"]["cents_abs_max"],
     "%.1f" % TD["resample"]["dur_abs_mean"], "%.1f" % TD["resample"]["dur_abs_max"]],
    ["Phase vocoder", "%.2f" % TD["pv"]["cents_abs_mean"], "%.2f" % TD["pv"]["cents_abs_max"],
     "%.1f" % TD["pv"]["dur_abs_mean"], "%.1f" % TD["pv"]["dur_abs_max"]],
    ["PV + formantes", "%.2f" % TD["pv_formant"]["cents_abs_mean"], "%.2f" % TD["pv_formant"]["cents_abs_max"],
     "%.1f" % TD["pv_formant"]["dur_abs_mean"], "%.1f" % TD["pv_formant"]["dur_abs_max"]],
], col_w=[CONTENT_W * 0.28] + [CONTENT_W * 0.18] * 4)
p("Los tres métodos afinan con error sub-audible (&lt; 5 cents). La diferencia está en la "
  "<b>duración</b>: el remuestreo se desvía en promedio %.0f %% (hasta ±50 %%), mientras el phase "
  "vocoder la conserva exactamente (0 %%)." % TD["resample"]["dur_abs_mean"])
fig("fig02_tono_duracion.png", caption="Fig. 2. Error de duración (izq.) y de afinación (der.) vs "
    "semitonos. El remuestreo deforma la duración; el phase vocoder la conserva. (PV y PV+formantes "
    "se superponen, pues comparten tono y duración.)")

h2("7.2 Preservación de formantes")
tabla([
    ["Método", "|desv. F1| medio [Hz]", "LSD medio [dB]", "centroide env. (ratio)"],
    ["Remuestreo", "%.0f" % FO["resample"]["F1_abs_shift_mean"], "%.1f" % FO["resample"]["lsd_mean_db"],
     "%.2f" % FO["resample"]["env_centroid_ratio_mean"]],
    ["Phase vocoder", "%.0f" % FO["pv"]["F1_abs_shift_mean"], "%.1f" % FO["pv"]["lsd_mean_db"],
     "%.2f" % FO["pv"]["env_centroid_ratio_mean"]],
    ["PV + formantes", "%.0f" % FO["pv_formant"]["F1_abs_shift_mean"], "%.1f" % FO["pv_formant"]["lsd_mean_db"],
     "%.2f" % FO["pv_formant"]["env_centroid_ratio_mean"]],
], col_w=[CONTENT_W * 0.28, CONTENT_W * 0.26, CONTENT_W * 0.22, CONTENT_W * 0.24])
p("La extensión <b>PV + formantes</b> es la única que mantiene el formante F1 cerca del original "
  "(desviación media %.0f Hz vs %.0f Hz del remuestreo), baja la LSD de la envolvente a %.1f dB "
  "(vs %.1f dB) y deja la razón de centroide de la envolvente en %.2f (~ 1 = sin corrimiento)."
  % (FO["pv_formant"]["F1_abs_shift_mean"], FO["resample"]["F1_abs_shift_mean"],
     FO["pv_formant"]["lsd_mean_db"], FO["resample"]["lsd_mean_db"],
     FO["pv_formant"]["env_centroid_ratio_mean"]))
fig("fig03_formantes.png", caption="Fig. 3. Envolvente espectral normalizada tras +5 semitonos (izq.) "
    "y posición del formante F1 vs semitonos (der.). PV + formantes (verde) sigue la línea original.")
fig("fig04_lsd.png", caption="Fig. 4. Distancia log-espectral de la envolvente respecto del original "
    "(menor = mejor preservación del timbre).")

h2("7.3 Caso real y síntesis integrada")
p("La Fig. 5 muestra un grano real de guitarra desplazado +7 semitonos por cada método (nótese el "
  "cambio de duración del remuestreo). La Fig. 6 y la Fig. 7 muestran la voz concatenativa completa "
  "“Pedal Miku” sintetizada con cada método. Los audios resultantes están en "
  "<font face='Courier'>outputs/</font> y embebidos en el notebook.")
fig("fig05_grano_real.png", caption="Fig. 5. Grano de guitarra desplazado +7 semitonos por método "
    "(espectrogramas).")
fig("fig06_sintesis_ondas.png", caption="Fig. 6. Formas de onda de la voz concatenativa por método.")
fig("fig07_sintesis_espectros.png", caption="Fig. 7. Espectrogramas de la voz sintetizada por método.")

h1("8. Discusión")
p("<b>Casos exitosos.</b> En desplazamientos moderados (-7 a +3 semitonos) sobre sonidos "
  "cuasi-estacionarios, el phase vocoder conserva perfectamente la duración y la extensión preserva el "
  "timbre, algo imposible con el remuestreo.")
p("<b>Casos de falla.</b> Para saltos grandes (>= +5–7 semitonos) la corrección de formantes se "
  "degrada (Fig. 3, der.): la envolvente desplazada se solapa poco con la original y la máscara, "
  "limitada a ±60 dB, no alcanza a restituirla. Además, el phase vocoder <b>emborrona los "
  "transitorios</b> (ataques de la guitarra) y aparece algo de <i>phasiness</i>, tal como advierte "
  "Dolson. El error de afinación del phase vocoder, aunque sub-audible, es mayor que el del remuestreo "
  "(interpolación de magnitud entre tramas).")
p("<b>Causas.</b> El phase vocoder asume estacionariedad dentro de la ventana; los transitorios y los "
  "grandes desplazamientos violan ese supuesto. La envolvente por cepstrum es global por grano y no "
  "captura bien formantes muy juntos.")

h1("9. Conclusiones")
p("Se reprodujo con éxito el phase vocoder de Dolson y se verificó su propiedad central: cambiar la "
  "duración sin alterar el tono. La extensión con preservación de formantes mejora claramente el "
  "timbre (LSD de %.1f a %.1f dB) conservando la ventaja de duración del phase vocoder, a costa de "
  "más cómputo y con fallas en saltos extremos. El proyecto se apoya íntegramente en herramientas "
  "clásicas del curso." % (FO["resample"]["lsd_mean_db"], FO["pv_formant"]["lsd_mean_db"]))
p("<b>Trabajo futuro:</b> <i>phase-locking</i> (Laroche–Dolson) para reducir phasiness, comparación "
  "con TD-PSOLA en el dominio del tiempo, y envolventes por LPC.")

h1("10. Código reutilizado y herramientas")
p("Se reutilizaron del proyecto previo <font face='Courier'>miku_pedal.ipynb</font> las utilidades de "
  "señal, el análisis de Fourier, la base de granos y la síntesis concatenativa (marcadas "
  "<font face='Courier'>[Reutilizado]</font> en <font face='Courier'>vocoder.py</font>). Son "
  "<b>desarrollo propio</b> de este proyecto: el phase vocoder, la preservación de formantes por "
  "cepstrum, todas las métricas y los experimentos. Bibliotecas: NumPy, SciPy, Matplotlib, soundfile, "
  "reportlab, nbformat. Se usó asistencia de IA como apoyo de programación y redacción, revisada por el autor.")

h1("Referencias")
p("[1] M. Dolson, “The Phase Vocoder: A Tutorial”, <i>Computer Music Journal</i>, vol. 10, n.º 4, "
  "pp. 14–27, 1986.")
p("[2] J. L. Flanagan y R. M. Golden, “Phase Vocoder”, <i>Bell System Technical Journal</i>, vol. 45, "
  "pp. 1493–1509, 1966.")
p("[3] J. Laroche y M. Dolson, “Improved phase vocoder time-scale modification of audio”, "
  "<i>IEEE Trans. Speech and Audio Processing</i>, vol. 7, n.º 3, pp. 323–332, 1999.")


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                            title="Informe Phase Vocoder PDDI", author="Francisco Pinto Abraham")
    doc.build(story)
    print("PDF escrito:", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    build()
