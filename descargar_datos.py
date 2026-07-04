# -*- coding: utf-8 -*-
"""
descargar_datos.py -- Obtiene el audio de guitarra usado como caso real.

Idempotente: si `data/guitarra.mp3` ya existe, no hace nada. Si no, intenta descargar
desde GitHub; si no hay red, copia la copia local del proyecto previo. La descarga es
E/S (andamiaje), no es materia del curso.
"""

import os
import shutil
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DEST = os.path.join(DATA, 'guitarra.mp3')

URL = 'https://raw.githubusercontent.com/k0ngan/audio/main/guitarra.mp3'
LOCAL_FALLBACK = os.path.join(HERE, '..', 'proyecto_1', 'audio_inputs', 'guitarra.mp3')


def main():
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1000:
        print('Ya existe:', DEST, '(%.0f KB)' % (os.path.getsize(DEST) / 1024))
        return DEST
    # 1) Intento de descarga desde GitHub
    try:
        print('Descargando desde', URL)
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp, open(DEST, 'wb') as fh:
            fh.write(resp.read())
        print('OK ->', DEST, '(%.0f KB)' % (os.path.getsize(DEST) / 1024))
        return DEST
    except Exception as exc:
        print('No se pudo descargar (%s). Uso copia local.' % exc)
    # 2) Copia local de respaldo
    if os.path.exists(LOCAL_FALLBACK):
        shutil.copyfile(LOCAL_FALLBACK, DEST)
        print('Copiado desde respaldo local ->', DEST)
        return DEST
    raise SystemExit('No hay audio disponible (sin red y sin copia local en %s).' % LOCAL_FALLBACK)


if __name__ == '__main__':
    main()
