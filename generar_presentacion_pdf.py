# -*- coding: utf-8 -*-
"""
generar_presentacion_pdf.py -- Presentación oral en PDF (diapositivas 16:9, estilo beamer).

Cubre el proyecto del artículo (phase vocoder de Dolson 1986 + extensión de formantes) y la aplicación
"Miku Stomp digital". Reutiliza las figuras de figuras/ y genera 2 figuras propias del stomp.
Salida: presentacion/presentacion_phase_vocoder_stomp.pdf

Texto WinAnsi-safe (sin glifos fuera de cp1252): se usan 'a', '~', 'pi', 'x' en vez de flechas/símbolos.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit

import vocoder as V
import stomp as S

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figuras")
PRES = os.path.join(HERE, "presentacion")
os.makedirs(FIG, exist_ok=True)
os.makedirs(PRES, exist_ok=True)
OUT = os.path.join(PRES, "presentacion_phase_vocoder_stomp.pdf")

# 16:9
PW, PH = 25.4 * cm, 14.2875 * cm
MX = 1.3 * cm
NAVY = colors.HexColor("#0b3d91")
TEAL = colors.HexColor("#11998e")
TEXT = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b7280")
LIGHT = colors.HexColor("#eef2fb")

R = json.load(open(os.path.join(HERE, "resultados.json"), encoding="utf-8")) \
    if os.path.exists(os.path.join(HERE, "resultados.json")) else {}


# ============================================================ figuras del stomp
def _spectro(ax, x, sr, title, fmax=2500):
    X = V.stft(x, n_fft=1024, hop=256)
    Sg = 20 * np.log10(np.abs(X).T + 1e-6)
    f = np.fft.rfftfreq(1024, 1.0 / sr); t = np.arange(X.shape[0]) * 256 / sr
    m = f <= fmax
    ax.pcolormesh(t, f[m], Sg[m], shading="auto", cmap="magma", vmin=Sg.max() - 70, vmax=Sg.max())
    ax.set_title(title, fontsize=11); ax.set_xlabel("tiempo [s]"); ax.set_ylabel("Hz")


def generar_figuras_stomp():
    sr = V.SR
    guitar = V.load_audio(os.path.join(HERE, "data", "guitarra.mp3"), sr)[:int(5 * sr)]
    vpath = os.path.join(HERE, "voces", "miku_voice.wav")
    voice = V.load_audio(vpath, sr) if os.path.exists(vpath) else V.synth_vowel(200.0, "a", 0.6)
    grain, vf0 = S.prep_voice_grain(voice, sr)
    times, f0, voiced = S.track_f0_pyin(guitar, sr)
    notes = S.segment_notes(times, f0, voiced)

    # fig f0 + notas sobre espectrograma
    X = V.stft(guitar, n_fft=1024, hop=256)
    Sg = 20 * np.log10(np.abs(X).T + 1e-6)
    ff = np.fft.rfftfreq(1024, 1.0 / sr); tt = np.arange(X.shape[0]) * 256 / sr
    m = ff <= 2000
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.pcolormesh(tt, ff[m], Sg[m], shading="auto", cmap="magma", vmin=Sg.max() - 70, vmax=Sg.max())
    ax.plot(times, f0, ".", ms=5, color="cyan", label="f0 (pYIN)")
    for (a, b, c) in notes:
        ax.hlines(c, a, b, color="white", lw=2.5)
    ax.set_ylim(0, 2000); ax.set_xlabel("tiempo [s]"); ax.set_ylabel("Hz")
    ax.set_title("Detección de notas: contorno de f0 (pYIN) y notas sobre el espectrograma")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "slide_stomp_f0.png"), dpi=130); plt.close(fig)

    # fig 3 modos
    labels = {"resample": "Remuestreo (robótico, como el pedal)",
              "pv": "Phase vocoder",
              "pv_formant": "PV + formantes (limpio)"}
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, mth in zip(ax, ["resample", "pv", "pv_formant"]):
        wet = S.miku_stomp_glide(times, f0, voiced, sr, grain, vf0, len(guitar), method=mth)
        _spectro(a, wet, sr, labels[mth])
    fig.suptitle("Miku Stomp: voz sintetizada por método (misma melodía de guitarra)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "slide_stomp_modos.png"), dpi=130); plt.close(fig)
    print("Figuras del stomp generadas (%d notas)." % len(notes))


# ============================================================ helpers de diapositivas
def _footer(c, n, total):
    c.setStrokeColor(colors.HexColor("#d0d7e2")); c.setLineWidth(0.6)
    c.line(MX, 0.95 * cm, PW - MX, 0.95 * cm)
    c.setFont("Helvetica", 8); c.setFillColor(MUTED)
    c.drawString(MX, 0.55 * cm, "Phase Vocoder + Miku Stomp  |  PDDI - UTEM 2026-1  |  F. Pinto")
    c.drawRightString(PW - MX, 0.55 * cm, "%d / %d" % (n, total))


def _titlebar(c, titulo):
    c.setFillColor(NAVY); c.rect(0, PH - 2.05 * cm, PW, 2.05 * cm, fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(0, PH - 2.16 * cm, PW, 0.11 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 20)
    c.drawString(MX, PH - 1.45 * cm, titulo)


def _bullets(c, bullets, x, y, w, fs=14, dy=0.92, lead=0.46):
    c.setFillColor(TEXT)
    for b in bullets:
        sub = b.startswith("  ")
        txt = b.strip()
        bx = x + (0.7 * cm if sub else 0)
        c.setFillColor(TEAL); c.setFont("Helvetica-Bold", fs)
        c.drawString(bx, y, "-" if sub else "•")
        c.setFillColor(TEXT); c.setFont("Helvetica", fs)
        lines = simpleSplit(txt, "Helvetica", fs, w - (bx - x) - 0.5 * cm)
        for ln in lines:
            c.drawString(bx + 0.5 * cm, y, ln); y -= lead * cm
        y -= (dy - lead) * cm
    return y


def _image_fit(c, path, x, y, w, h):
    iw, ih = PILImage.open(path).size
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    c.drawImage(path, x + (w - dw) / 2, y + (h - dh) / 2, dw, dh,
                preserveAspectRatio=True, mask="auto")


# ============================================================ construir deck
def build():
    if not os.path.exists(os.path.join(FIG, "fig02_tono_duracion.png")):
        import generar_resultados
        generar_resultados.main()
    generar_figuras_stomp()

    td = R.get("tono_duracion_resumen", {})
    fo = R.get("formantes_resumen", {})
    lsd_res = fo.get("resample", {}).get("lsd_mean_db", 8.6)
    lsd_pvf = fo.get("pv_formant", {}).get("lsd_mean_db", 1.6)
    dur_res = td.get("resample", {}).get("dur_abs_max", 50.0)

    c = canvas.Canvas(OUT, pagesize=(PW, PH))
    TOTAL = 12
    n = [0]

    def newpage_footer():
        n[0] += 1
        _footer(c, n[0], TOTAL)
        c.showPage()

    # 1 PORTADA
    c.setFillColor(NAVY); c.rect(0, 0, PW, PH, fill=1, stroke=0)
    c.setFillColor(TEAL); c.rect(0, PH * 0.5 - 0.06 * cm, PW, 0.12 * cm, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 27)
    c.drawCentredString(PW / 2, PH * 0.62, "Cambiar el tono sin alterar la duración")
    c.setFont("Helvetica", 16)
    c.drawCentredString(PW / 2, PH * 0.62 - 0.95 * cm, "Phase Vocoder (Dolson 1986) + Miku Stomp digital")
    c.setFont("Helvetica", 12.5); c.setFillColor(colors.HexColor("#cfe0ff"))
    c.drawCentredString(PW / 2, PH * 0.40, "Procesamiento Digital de Señales e Imágenes (INFB6063) - UTEM 2026-1")
    c.drawCentredString(PW / 2, PH * 0.40 - 0.8 * cm, "Francisco Alejandro Pinto Abraham  -  RUT 21.571.239-7")
    c.setFont("Helvetica-Oblique", 11)
    c.drawCentredString(PW / 2, PH * 0.18, "Artículo: M. Dolson, \"The Phase Vocoder: A Tutorial\", CMJ 10(4), 1986")
    c.setFont("Helvetica", 10); c.setFillColor(TEAL)
    c.drawCentredString(PW / 2, PH * 0.18 - 0.7 * cm, "github.com/k0ngan/mikustromp")
    n[0] += 1; c.showPage()

    # 2 PROBLEMA
    _titlebar(c, "1. El problema")
    _bullets(c, [
        "Subir o bajar el tono de un sonido (afinar, transponer, armonizar).",
        "La vía ingenua (remuestreo) sube el tono PERO:",
        "  cambia la duración (acorta o alarga el audio),",
        "  corre los formantes -> efecto \"ardilla\".",
        "Objetivo: desacoplar tono, duración y timbre con DSP clásico.",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=15)
    newpage_footer()

    # 3 ARTICULO
    _titlebar(c, "2. El artículo: Dolson (1986)")
    _bullets(c, [
        "Analiza la señal con la STFT: por banda, separa magnitud y FASE.",
        "La fase da la FRECUENCIA INSTANTÁNEA (diferencia de fase entre tramas, mod 2 pi).",
        "Time-stretch: re-sintetizar con un salto distinto, acumulando la fase -> cambia",
        "  la duración sin tocar el tono.",
        "Pitch-shift = time-stretch + remuestreo.",
        "Limitaciones que reporta: phasiness y emborronado de transitorios.",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=14.5)
    newpage_footer()

    # 4 RELACION CURSO
    _titlebar(c, "3. Relación con el curso")
    _bullets(c, [
        "STFT y enventanado Hann  ->  Unit 2 - L1",
        "Espectro, magnitud y fase de la DFT  ->  Unit 1 - L2/L3",
        "Frecuencia instantánea (fase, unwrapping)  ->  Unit 1 - L2",
        "Síntesis por superposición (overlap-add)  ->  Unit 2 - L1/L2",
        "Pitch-shift = stretch + remuestreo  ->  Unit 1 - L3",
        "Preservación de formantes (máscara H[k]) + cepstrum  ->  Unit 2 - L3",
        "Todo DSP clásico: sin deep learning / IA generativa.",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=14)
    newpage_footer()

    # 5 REPRODUCCION (fig)
    _titlebar(c, "4. Reproducción: duración cambia, tono no")
    _image_fit(c, os.path.join(FIG, "fig01_reproduccion_pv.png"), MX, 1.2 * cm, PW - 2 * MX, PH - 3.6 * cm)
    newpage_footer()

    # 6 EXTENSION
    _titlebar(c, "5. Extensión propia: PV + formantes")
    _bullets(c, [
        "El pitch-shift básico mueve los formantes junto con los armónicos.",
        "Idea: re-imponer la ENVOLVENTE espectral original tras subir el tono.",
        "Se estima la envolvente por CEPSTRUM (lifter de baja cuefrencia).",
        "Máscara de corrección: H[k] = env_original / env_desplazada (limitada a +-60 dB).",
        "Resultado: los armónicos suben, pero los formantes (timbre) quedan fijos.",
        "Es filtrado en frecuencia (Unit 2 - L3) guiado por cepstrum. Sin ML.",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=14.5)
    newpage_footer()

    # 7 RESULTADOS tono/duración (fig + nota)
    _titlebar(c, "6. Resultados: tono y duración")
    _image_fit(c, os.path.join(FIG, "fig02_tono_duracion.png"), MX, 2.2 * cm, PW - 2 * MX, PH - 4.6 * cm)
    c.setFont("Helvetica-Bold", 12.5); c.setFillColor(NAVY)
    c.drawCentredString(PW / 2, 1.4 * cm,
                        "Remuestreo: hasta +-%.0f%% de error de duración  |  Phase vocoder: 0%% (afinación < 5 cents)"
                        % dur_res)
    newpage_footer()

    # 8 RESULTADOS formantes (dos figuras lado a lado)
    _titlebar(c, "7. Resultados: formantes")
    half = (PW - 2 * MX - 0.5 * cm) / 2
    _image_fit(c, os.path.join(FIG, "fig03_formantes.png"), MX, 2.2 * cm, half, PH - 4.6 * cm)
    _image_fit(c, os.path.join(FIG, "fig04_lsd.png"), MX + half + 0.5 * cm, 2.2 * cm, half, PH - 4.6 * cm)
    c.setFont("Helvetica-Bold", 12.5); c.setFillColor(NAVY)
    c.drawCentredString(PW / 2, 1.4 * cm,
                        "Sólo PV + formantes mantiene F1 y baja la distorsión de la envolvente: ~%.1f dB -> ~%.1f dB"
                        % (lsd_res, lsd_pvf))
    newpage_footer()

    # 9 APLICACION stomp
    _titlebar(c, "8. Aplicación: Miku Stomp digital")
    _bullets(c, [
        "Replica del pedal Korg Miku Stomp en Python (DSP puro).",
        "Flujo: guitarra -> detectar la nota (pYIN) -> afinar una voz a esa nota -> mezclar.",
        "Usa nuestro pitch-shift CON formantes: la voz mantiene su timbre al seguir la melodía.",
        "Mejor que el pedal: el Korg corre los formantes (ardilla); aquí no.",
        "Voz real incluida (Vocaloid 4 CyberDiva); pYIN es DSP clásico, no una red neuronal.",
        "Cuaderno autocontenido para Google Colab (subir audio -> procesar -> descargar).",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=14)
    newpage_footer()

    # 10 stomp f0 (fig)
    _titlebar(c, "9. Miku Stomp: detección de notas (pYIN)")
    _image_fit(c, os.path.join(FIG, "slide_stomp_f0.png"), MX, 1.2 * cm, PW - 2 * MX, PH - 3.6 * cm)
    newpage_footer()

    # 11 stomp modos (fig)
    _titlebar(c, "10. Miku Stomp: 3 modos (robótico vs limpio)")
    _image_fit(c, os.path.join(FIG, "slide_stomp_modos.png"), MX, 1.4 * cm, PW - 2 * MX, PH - 3.8 * cm)
    newpage_footer()

    # 12 DISCUSION / CONCLUSIONES
    _titlebar(c, "11. Discusión y conclusiones")
    _bullets(c, [
        "Funciona: desplazamientos moderados (-7 a +3 semitonos) en sonidos cuasi-estacionarios.",
        "Falla: saltos grandes (>= +5/+7) degradan la corrección de formantes; transitorios borrosos.",
        "Conceptos clave: STFT, fase/frecuencia instantánea, overlap-add, cepstrum, muestreo.",
        "La extension MEJORÓ el timbre (LSD ~%.1f -> ~%.1f dB) conservando la duración." % (lsd_res, lsd_pvf),
        "Trabajo futuro: phase-locking (Laroche-Dolson), TD-PSOLA, polifonía/RVC (solo mención).",
        "Código, informe, audios y cuaderno: github.com/k0ngan/mikustromp",
    ], MX, PH - 3.2 * cm, PW - 2 * MX, fs=13.5)
    newpage_footer()

    c.save()
    print("PDF de diapositivas escrito:", OUT, "(%d paginas, %.0f KB)" % (n[0], os.path.getsize(OUT) / 1024))


if __name__ == "__main__":
    build()
