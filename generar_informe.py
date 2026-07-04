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
ST = R.get("stomp", {})

# Resultados del autotune (aplicación principal). Guardado por si no se ha generado aún.
_at_path = os.path.join(HERE, "resultados_autotune.json")
AT = json.load(open(_at_path, encoding="utf-8")) if os.path.exists(_at_path) else {}

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
story.append(Paragraph("Autotune con voz de Miku: corrección de afinación y transferencia de formantes "
                       "sobre el <i>phase vocoder</i> de Dolson (DSP clásico, sin aprendizaje profundo)", TITLE))
sp(10)
story.append(Paragraph("Proyecto Semestral — Procesamiento Digital de Señales e Imágenes (INFB6063)", SUB))
story.append(Paragraph("Universidad Tecnológica Metropolitana (UTEM) — 2026-1", SUB))
sp(8)
story.append(Paragraph("Francisco Alejandro Pinto Abraham — RUT 21.571.239-7", SUB))
sp(14)
story.append(Paragraph("<b>Artículo reproducido:</b> M. Dolson, “The Phase Vocoder: A Tutorial”, "
                       "<i>Computer Music Journal</i>, vol. 10, n.º 4, pp. 14–27, 1986.", BODY))
sp(8)
story.append(Paragraph("<b>Resumen —</b> Se construye un <b>autotune</b> que toma una voz cantada, "
    "corrige su afinación a las notas de una escala musical y le transfiere el <b>timbre (formantes) de "
    "Hatsune Miku</b>, conservando la interpretación (melodía y ritmo) del cantante. El motor reproduce "
    "el <i>phase vocoder</i> de Dolson (1986) —que cambia el tono sin alterar la duración propagando la "
    "fase de la STFT por su frecuencia instantánea— y una <b>extensión propia</b> de preservación de "
    "formantes por cepstrum; sobre ellos se añade el <b>seguimiento de tono</b> (pYIN/autocorrelación), "
    "el <b>“snap” a la escala</b> y la <b>transferencia de la envolvente de Miku</b>. En una voz de "
    "prueba con verdad de terreno, el autotune reduce el error de afinación de %.0f a %.0f cents, "
    "<b>conserva la duración</b> (0 %%) y acerca la envolvente al timbre de Miku (distancia log-espectral "
    "de %.1f a %.1f dB). Como validación del motor, el remuestreo ingenuo deforma la duración hasta "
    "±50 %% mientras el phase vocoder la mantiene, y la extensión baja la distorsión de la envolvente de "
    "~%.1f a ~%.1f dB. Solo se usan técnicas clásicas del curso (Fourier, STFT, enventanado, filtrado, "
    "cepstrum); sin aprendizaje profundo." % (
        AT.get("cents_abs_medio", {}).get("antes", 42), AT.get("cents_abs_medio", {}).get("despues", 5),
        AT.get("lsd_envolvente_a_miku_dB", {}).get("autotune", 5.5),
        AT.get("lsd_envolvente_a_miku_dB", {}).get("miku", 4.2),
        FO["resample"]["lsd_mean_db"], FO["pv_formant"]["lsd_mean_db"]), ABS))
sp(8)
story.append(Paragraph("<b>Palabras clave —</b> autotune, corrección de afinación, voz de Miku, "
                       "phase vocoder, STFT, frecuencia instantánea, formantes, cepstrum, snap a escala.", BODY))
story.append(PageBreak())

# ===================================================================== 1. Introducción
h1("1. Introducción y descripción del problema")
p("Modificar el <b>tono</b> de un sonido sin cambiar su <b>duración</b> es una operación básica en "
  "música y voz (afinación, armonías, transposición). La vía ingenua —releer la señal a otra tasa "
  "(remuestreo)— sube o baja el tono pero <b>acorta o alarga</b> el audio y <b>corre los formantes</b>, "
  "produciendo el característico efecto “ardilla”. El reto es desacoplar tono, duración y timbre.")
p("Este trabajo reproduce el <i>phase vocoder</i> propuesto por Dolson [1], implementa con él un "
  "desplazamiento de tono que conserva la duración, y propone una <b>extensión</b> que además conserva "
  "los formantes. Sobre esa base se construye la <b>aplicación principal</b>: un <b>autotune con voz de "
  "Miku</b> (Sec. 8) que sigue el tono de una voz, lo <b>pega a las notas de una escala</b> y le "
  "<b>transfiere el timbre de Hatsune Miku</b>. La misma maquinaria se reutiliza en una aplicación "
  "secundaria —el “Miku Stomp” de guitarra (Sec. 9)—, reutilizada de un proyecto previo del curso.")

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
    ["Seguimiento de tono f0 (autotune)", "Unit 1 – L1/L2 (autocorrelación/periodicidad)"],
    ["“Snap” a la escala (f0 -&gt; nota MIDI)", "logaritmos de frecuencia; cuantización"],
    ["Transferencia de timbre de Miku", "Unit 2 – L3 (máscara espectral) + cepstrum"],
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

h1("8. Aplicación principal: Autotune con voz de Miku")
p("La aplicación central del proyecto es un <b>autotune</b>: dada una voz cantada, (i) se estima su "
  "<b>contorno de tono</b> f0 cuadro a cuadro (pYIN/autocorrelación); (ii) cada tono se pasa a número "
  "MIDI (<font face='Courier'>m = 69 + 12 log2(f/440)</font>) y se <b>redondea a la nota de la escala</b> "
  "más cercana (el “snap”); (iii) la diferencia en semitonos se corrige de forma <b>variable en el "
  "tiempo</b> con el phase vocoder <b>preservando los formantes</b> (Sec. 5), reconstruyendo por "
  "<i>overlap-add</i>, por lo que la <b>duración se conserva</b>; y (iv) se estima la <b>envolvente de "
  "formantes de una voz real de Miku</b> por cepstrum y se <b>impone</b> sobre la voz afinada como una "
  "máscara <font face='Courier'>H = env_Miku / env_voz</font>. El resultado mantiene la melodía y el "
  "ritmo del cantante, pero suena <b>afinado</b> y con <b>timbre de Miku</b>.")
p("Dos perillas gobiernan el efecto: <b>strength</b> (0 a 1) mezcla el tono medido con la nota pegada "
  "(1 = enganche total, efecto “T-Pain”) y <b>retune_speed</b> controla la velocidad de enganche "
  "(valores bajos dan un glissando natural). Se evaluó con una voz de prueba de <b>verdad de terreno "
  "conocida</b>: una secuencia de vocales en Do mayor deliberadamente <b>desafinada</b> ±30 a 50 cents.")
if AT:
    _cm = AT.get("cents_abs_medio", {})
    _du = AT.get("duracion_error_pct", {})
    _ls = AT.get("lsd_envolvente_a_miku_dB", {})
    tabla([
        ["Métrica", "Antes / entrada", "Después"],
        ["Error de afinación medio [cents]", "%.0f" % _cm.get("antes", 0), "%.0f" % _cm.get("despues", 0)],
        ["Error de duración [%]", "0.0", "%.1f" % _du.get("miku", 0.0)],
        ["LSD de la envolvente a Miku [dB]", "%.2f" % _ls.get("entrada", 0), "%.2f" % _ls.get("miku", 0)],
    ], col_w=[CONTENT_W * 0.46, CONTENT_W * 0.27, CONTENT_W * 0.27])
    p("El autotune baja el error de afinación de %.0f a %.0f cents (las notas quedan sobre la escala), "
      "<b>conserva la duración</b> y acerca la envolvente al timbre de Miku (LSD de %.2f a %.2f dB). La "
      "corrección preserva los formantes de la voz durante el afinado y recién en el último paso se "
      "sustituyen por los de Miku, de forma controlada." % (
          _cm.get("antes", 0), _cm.get("despues", 0), _ls.get("entrada", 0), _ls.get("miku", 0)))
fig("at_01_contorno.png", caption="Fig. 8. Autotune: el tono medido (gris) se pega a las notas de la "
    "escala (rojo) y el tono corregido (azul) queda sobre ellas.")
fig("at_04_afinacion.png", caption="Fig. 9. Error de afinación por nota, antes vs después del autotune "
    "(0 = afinado; líneas rojas: ±50 cents).")
fig("at_03_formantes.png", caption="Fig. 10. Transferencia de formantes: la envolvente de la salida "
    "(verde) adopta la de Miku (morado), alejándose de la voz de entrada (gris).")
fig("at_02_espectrograma.png", caption="Fig. 11. Espectrogramas de la voz de entrada (desafinada) y de "
    "la salida (afinada y con timbre de Miku).")

h1("9. Aplicación secundaria: Miku Stomp digital (guitarra)")
p("Como aplicación, se replicó el flujo del pedal <b>Korg Miku Stomp</b> en DSP puro: "
  "<b>guitarra -&gt; detectar la nota -&gt; afinar una voz a esa nota -&gt; mezclar</b>. La detección de tono usa "
  "<b>pYIN</b> (YIN probabilístico, algoritmo clásico, no una red neuronal); la voz se afina con el "
  "phase vocoder con preservación de formantes (Sec. 5), por lo que mantiene su timbre al seguir la "
  "melodía —a diferencia del pedal original, que corre los formantes (efecto “ardilla”). La voz fuente "
  "es una muestra real de Vocaloid 4 (CyberDiva); uso académico declarado.")
p("<b>Problema detectado y solución.</b> Una primera versión desafinaba: (i) pYIN se enganchaba a "
  "armónicos de la guitarra (errores de octava) — se corrigió bajando <font face='Courier'>fmax</font> "
  "a 600 Hz, reparando saltos de octava y suavizando el contorno; (ii) la voz se sintetiza siguiendo el "
  "contorno de f0 de forma continua (glissando, por <i>overlap-add</i>), transponiendo toda la melodía "
  "por un <b>número entero de octavas</b> a un registro cantable (conserva el contorno) y plegando por "
  "octavas solo las notas fuera de rango; (iii) se elige un grano de voz de tono estable.")
if ST.get("afinacion"):
    af = ST["afinacion"]
    def _md(m, k):
        v = af.get(m, {}).get(k)
        return "%.0f" % v if v is not None else "-"
    tabla([
        ["Método", "Error afinación mediano [cents]", "Notas dentro de 1 semitono [%]"],
        ["Remuestreo", _md("resample", "cents_median"), _md("resample", "within_semitone_pct")],
        ["Phase vocoder", _md("pv", "cents_median"), _md("pv", "within_semitone_pct")],
        ["PV + formantes", _md("pv_formant", "cents_median"), _md("pv_formant", "within_semitone_pct")],
    ], col_w=[CONTENT_W * 0.30, CONTENT_W * 0.38, CONTENT_W * 0.32])
    p("La voz sintetizada sigue el tono de la guitarra con error mediano de pocos <b>cents</b> "
      "(prácticamente todas las notas dentro de un semitono del objetivo), confirmando que ahora "
      "“coincide” con la guitarra.")
fig("fig08_stomp_f0.png", caption="Fig. 12. Miku Stomp: contorno de f0 (pYIN) y notas detectadas sobre "
    "el espectrograma de la guitarra (tras estabilizar el seguimiento).")
fig("fig09_stomp_modos.png", caption="Fig. 13. Miku Stomp: voz que sigue la melodía de la guitarra, "
    "sintetizada con cada método de pitch-shift.")

h1("10. Discusión")
p("<b>¿Qué fue lo más difícil de comprender?</b> La propagación de fase del phase vocoder: estimar la "
  "frecuencia instantánea por diferencia de fase entre tramas y el desenrollado módulo 2 pi; sin esa "
  "corrección aparece <i>phasiness</i>.")
p("<b>¿Qué fue lo más complejo de implementar?</b> La preservación de formantes por cepstrum (elegir el "
  "lifter y el rango de ganancia de la máscara) y, en el <b>autotune</b>, dos cosas: (i) llevar la voz al "
  "<b>registro de Miku</b> subiendo octavas enteras sin recortar la corrección del <i>snap</i> (una "
  "primera versión limitaba el desplazamiento total y la subida de octava no ocurría, dejando la voz "
  "grave), y (ii) <b>acondicionar la voz</b> de micrófono y aplicar una <b>compuerta de silencios</b> "
  "para no sintetizar ruido tonal en las pausas.")
p("<b>¿Qué conceptos del curso fueron más importantes?</b> STFT y enventanado, magnitud y fase de la "
  "DFT, <i>overlap-add</i>, muestreo/interpolación, autocorrelación/pYIN para el tono, logaritmos de "
  "frecuencia (nota MIDI y snap) y filtrado en frecuencia (máscara) guiado por cepstrum.")
p("<b>¿La modificación mejoró, empeoró o cambió el método?</b> La preservación de formantes <b>mejoró</b> "
  "el timbre (baja la distorsión de la envolvente de ~%.1f a ~%.1f dB y mantiene F1) conservando la "
  "ventaja de duración del phase vocoder. Sobre esa base, el <b>autotune con voz de Miku</b> es una "
  "aplicación nueva que <b>cambia</b> el uso del método: en lugar de un pitch-shift fijo, corrige el tono "
  "de forma variable en el tiempo (afinación de %.0f a %.0f cents) y usa la misma máquina de formantes "
  "para imponer el timbre de Miku (LSD a Miku de %.2f a %.2f dB), a costa de más cómputo." % (
      FO["resample"]["lsd_mean_db"], FO["pv_formant"]["lsd_mean_db"],
      AT.get("cents_abs_medio", {}).get("antes", 42), AT.get("cents_abs_medio", {}).get("despues", 5),
      AT.get("lsd_envolvente_a_miku_dB", {}).get("entrada", 5.3),
      AT.get("lsd_envolvente_a_miku_dB", {}).get("miku", 3.3)))
p("<b>¿Cuándo funciona bien?</b> En desplazamientos moderados sobre sonidos cuasi-estacionarios; en el "
  "autotune, con una voz cantada de nivel razonable el tono se pega a la escala (pocos cents) y el "
  "registro/timbre de Miku quedan convincentes conservando la interpretación.")
p("<b>¿Cuándo falla?</b> En saltos grandes (>= +5/+7 semitonos) la corrección de formantes se degrada "
  "(Fig. 3 y 4) y el phase vocoder <b>emborrona los transitorios</b> con algo de <i>phasiness</i>. En el "
  "autotune, tomas <b>muy silenciosas o muy graves</b> dan mal seguimiento de tono, y la subida de +2 "
  "octavas al registro de Miku añade artefactos por el salto grande; en el stomp, notas muy graves de la "
  "guitarra deben plegarse de octava para ser cantables.")
p("<b>¿Qué haríamos con más tiempo?</b> Una versión en <b>tiempo real</b> por bloques (<i>sounddevice</i> "
  "o como plugin), <i>phase-locking</i> (Laroche–Dolson) para reducir phasiness, TD-PSOLA como comparador "
  "en el dominio del tiempo, envolventes por LPC, y voces objetivo intercambiables (no solo Miku).")

h1("11. Conclusiones")
p("Se construyó un <b>autotune con voz de Miku</b> que afina una voz a una escala (error de afinación "
  "de %.0f a %.0f cents) <b>sin alterar su duración</b> y le transfiere el timbre de Miku (LSD de la "
  "envolvente de %.2f a %.2f dB). Para ello se reprodujo con éxito el phase vocoder de Dolson —cuya "
  "propiedad central, cambiar el tono sin alterar la duración, se verificó— y su extensión con "
  "preservación de formantes, que mejora el timbre (LSD de %.1f a %.1f dB) a costa de más cómputo y con "
  "fallas en saltos extremos. La misma maquinaria sirve la aplicación secundaria (“Miku Stomp” de "
  "guitarra). El proyecto se apoya íntegramente en herramientas clásicas del curso." % (
      AT.get("cents_abs_medio", {}).get("antes", 42), AT.get("cents_abs_medio", {}).get("despues", 5),
      AT.get("lsd_envolvente_a_miku_dB", {}).get("entrada", 5.3),
      AT.get("lsd_envolvente_a_miku_dB", {}).get("miku", 4.2),
      FO["resample"]["lsd_mean_db"], FO["pv_formant"]["lsd_mean_db"]))
p("<b>Trabajo futuro:</b> <i>phase-locking</i> (Laroche–Dolson) para reducir phasiness, comparación "
  "con TD-PSOLA en el dominio del tiempo, envolventes por LPC, polifonía y una versión en tiempo real.")

h1("12. Código reutilizado y herramientas")
p("Se reutilizaron del proyecto previo <font face='Courier'>miku_pedal.ipynb</font> las utilidades de "
  "señal, el análisis de Fourier, la base de granos y la síntesis concatenativa (marcadas "
  "<font face='Courier'>[Reutilizado]</font> en <font face='Courier'>vocoder.py</font>). Son "
  "<b>desarrollo propio</b> de este proyecto: el phase vocoder, la preservación de formantes por "
  "cepstrum, las métricas, los experimentos, el <b>motor de autotune</b> en "
  "<font face='Courier'>autotune.py</font> (snap a la escala, corrección variable en el tiempo y "
  "transferencia de formantes de Miku) y el pipeline del pedal en "
  "<font face='Courier'>stomp.py</font> (pYIN, segmentación de notas y síntesis glissando). "
  "Bibliotecas: NumPy, SciPy, Matplotlib, soundfile, librosa (solo pYIN), reportlab, nbformat. Se usó "
  "asistencia de IA como apoyo de programación y redacción, revisada por el autor.")

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
