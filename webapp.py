#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR — drag-and-drop web demo.

A tiny Flask app: open it in a browser, drop a car photo, and see the detected
plate boxed with its reading, country, class, and state. Good for showing the
system without the command line.

Run:
    ./venv/bin/pip install flask           # one time
    ./venv/bin/python webapp.py
    # then open http://127.0.0.1:5000

This reuses the exact same pipeline as recognize_trained.py (fine-tuned OCR +
the sudan_plate interpreter), so what you see here matches the CLI.
"""

from __future__ import annotations

import base64
import io
import os

from flask import Flask, request, render_template_string

HERE = os.path.dirname(os.path.abspath(__file__))
OCR_MODEL = os.path.join(HERE, "models", "sudan_ocr.onnx")
PLATE_CFG = os.path.join(HERE, "models", "sudan_plate.yaml")

app = Flask(__name__)
_alpr = None  # lazy-loaded on first request so startup is instant


def get_alpr():
    """Load the detector + fine-tuned OCR once and cache it."""
    global _alpr
    if _alpr is None:
        from fast_alpr import ALPR
        _alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model_path=OCR_MODEL,
            ocr_config_path=PLATE_CFG,
        )
    return _alpr


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sudan ALPR</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto;
           padding: 0 1rem; line-height: 1.5; }
    h1 { display: flex; align-items: center; gap: .5rem; }
    .drop { border: 2px dashed #888; border-radius: 12px; padding: 2.5rem;
            text-align: center; cursor: pointer; transition: .15s; }
    .drop:hover, .drop.over { border-color: #2e7d32; background: rgba(46,125,50,.06); }
    .plate { border: 1px solid #8884; border-radius: 10px; padding: .8rem 1rem;
             margin: .6rem 0; }
    .plate .text { font-size: 1.4rem; font-weight: 700; letter-spacing: .04em; }
    .muted { color: #888; font-size: .9rem; }
    img.result { max-width: 100%; border-radius: 10px; margin-top: 1rem; }
    .pill { display: inline-block; background: #2e7d3222; color: #2e7d32;
            border-radius: 999px; padding: .1rem .6rem; font-size: .85rem; }
  </style>
</head>
<body>
  <h1>🇸🇩 Sudan ALPR</h1>
  <p class="muted">Drop a car photo to detect and read the plate.</p>

  <form id="f" method="post" enctype="multipart/form-data">
    <label class="drop" id="drop">
      <input id="file" type="file" name="image" accept="image/*" hidden>
      <strong>Click or drop an image here</strong>
      <div class="muted">JPG / PNG</div>
    </label>
  </form>

  {% if results is not none %}
    <h2>Result</h2>
    {% if results %}
      {% for p in results %}
        <div class="plate">
          <div class="text">{{ p.text or "—" }}</div>
          <div>
            {% if p.is_sudan %}🇸🇩{% else %}🌐{% endif %}
            {{ p.country }}{% if p.locality %} / {{ p.locality }}{% endif %}
            <span class="pill">{{ p.plate_class_en }}</span>
          </div>
          <div class="muted">country {{ p.country_pct }}% · detect {{ p.detect_pct }}%</div>
        </div>
      {% endfor %}
    {% else %}
      <p class="muted">No plate detected in that image.</p>
    {% endif %}
    {% if annotated %}<img class="result" src="data:image/jpeg;base64,{{ annotated }}">{% endif %}
  {% endif %}

  <script>
    const drop = document.getElementById('drop'),
          file = document.getElementById('file'),
          form = document.getElementById('f');
    file.addEventListener('change', () => { if (file.files.length) form.submit(); });
    ['dragover','dragenter'].forEach(e => drop.addEventListener(e, ev => {
      ev.preventDefault(); drop.classList.add('over'); }));
    ['dragleave','drop'].forEach(e => drop.addEventListener(e, ev => {
      ev.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', ev => { file.files = ev.dataTransfer.files; form.submit(); });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET" or "image" not in request.files:
        return render_template_string(PAGE, results=None, annotated=None)

    upload = request.files["image"]
    if not upload or not upload.filename:
        return render_template_string(PAGE, results=None, annotated=None)

    import cv2
    import numpy as np
    from sudan_plate import interpret

    data = np.frombuffer(upload.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return render_template_string(PAGE, results=[], annotated=None)

    alpr = get_alpr()
    drawn = alpr.draw_predictions(frame)

    results = []
    for r in drawn.results:
        info = interpret(r.ocr.text if r.ocr else "")
        if info.is_sudan and info.plate_class == "private":
            locality = info.state
        elif info.is_sudan:
            locality = info.plate_class_en
        else:
            locality = ""
        results.append({
            "text": info.text,
            "is_sudan": info.is_sudan,
            "country": info.country,
            "plate_class_en": info.plate_class_en,
            "locality": locality,
            "country_pct": round(info.country_confidence * 100),
            "detect_pct": round(float(r.detection.confidence) * 100),
        })

    ok, buf = cv2.imencode(".jpg", drawn.image)
    annotated = base64.b64encode(buf).decode() if ok else None
    return render_template_string(PAGE, results=results, annotated=annotated)


if __name__ == "__main__":
    print("Sudan ALPR web demo →  http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
