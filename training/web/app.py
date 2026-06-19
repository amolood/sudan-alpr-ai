#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sudan ALPR — all-in-one web studio.

Screens (single page, three tabs):
    رفع     /  upload car photos -> auto-detect + crop plates into the pool
    توسيم   /  label each plate (model pre-fills its guess; you confirm/fix)
    تدريب   /  fine-tune on the labelled set, live log + held-out accuracy %

The goal is the improvement loop: upload more → label → retrain → accuracy rises.

Run:
    ./venv/bin/python training/web/app.py
    open http://127.0.0.1:8000
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import threading
import time

from flask import Flask, jsonify, request, send_file, Response

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "training", "dataset")
PLATES_DIR = os.path.join(DS, "good_plates")       # labelling pool
UPLOAD_DIR = os.path.join(DS, "uploads")           # raw uploaded car photos
CSV_PATH = os.path.join(DS, "labels.csv")
SCRIPTS = os.path.join(ROOT, "training", "scripts")
MODELS = os.path.join(ROOT, "models")
PY = os.path.join(ROOT, "venv", "bin", "python")
FPO = os.path.join(ROOT, "venv", "bin", "fast-plate-ocr")

os.makedirs(PLATES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

app = Flask(__name__)

_alpr = None
_cv2 = None
FMT = re.compile(r"^[0-9]{1,2}[A-Z]{2,3}[0-9]{2,5}$")


def _engine():
    global _alpr, _cv2
    if _alpr is None:
        import cv2
        from fast_alpr import ALPR
        import sys
        sys.path.insert(0, ROOT)
        _cv2 = cv2
        _alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end",
                     ocr_model="cct-xs-v2-global-model")
    return _alpr, _cv2


def _guess(path):
    alpr, cv2 = _engine()
    from recognize import read_sudan_plate
    img = cv2.imread(path)
    if img is None:
        return ""
    text, _ = read_sudan_plate(img, alpr.ocr, cv2)
    return text


# ---- labels ----------------------------------------------------------------
def load_labels():
    out = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[os.path.basename(r["image_path"])] = r["plate_text"]
    return out


def save_labels(labels):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text"])
        for name, text in labels.items():
            w.writerow([os.path.join(PLATES_DIR, name), text])


def plate_list():
    return sorted(f for f in os.listdir(PLATES_DIR) if f.lower().endswith(".png"))


# ---- routes: status & images ----------------------------------------------
@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/plates")
def api_plates():
    names = plate_list()
    labels = load_labels()
    items = [{"name": n, "label": labels.get(n, "")} for n in names]
    return jsonify({"plates": items, "labelled": len(labels), "total": len(names)})


@app.route("/api/image/<name>")
def api_image(name):
    path = os.path.join(PLATES_DIR, os.path.basename(name))
    if not os.path.exists(path):
        return "not found", 404
    return send_file(path, mimetype="image/png")


@app.route("/api/guess/<name>")
def api_guess(name):
    return jsonify({"guess": _guess(os.path.join(PLATES_DIR, os.path.basename(name)))})


@app.route("/api/save", methods=["POST"])
def api_save():
    d = request.get_json(force=True)
    name = os.path.basename(d["name"])
    text = (d.get("text") or "").strip().upper()
    labels = load_labels()
    if text:
        labels[name] = text
    else:
        labels.pop(name, None)
    save_labels(labels)
    return jsonify({"ok": True, "labelled": len(labels)})


# ---- routes: upload --------------------------------------------------------
_upload_state = {"running": False, "log": [], "added": 0}


def _process_uploads(paths):
    _upload_state.update(running=True, log=[], added=0)
    alpr, cv2 = _engine()
    existing = set(plate_list())
    n = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            _upload_state["log"].append(f"تعذّر فتح {os.path.basename(p)}")
            continue
        dets = [d for d in alpr.detector.predict(img) if float(d.confidence) > 0.5]
        if not dets:
            _upload_state["log"].append(f"لا لوحة في {os.path.basename(p)}")
            continue
        stem = re.sub(r"[^A-Za-z0-9]", "_", os.path.splitext(os.path.basename(p))[0])
        for i, det in enumerate(dets):
            bb = det.bounding_box
            x1, y1 = max(0, int(bb.x1)), max(0, int(bb.y1))
            x2, y2 = int(bb.x2), int(bb.y2)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[1] < 90:
                continue
            name = f"up_{stem}_{i}.png"
            k = 1
            while name in existing:
                name = f"up_{stem}_{i}_{k}.png"
                k += 1
            cv2.imwrite(os.path.join(PLATES_DIR, name), crop)
            existing.add(name)
            n += 1
        _upload_state["log"].append(f"✓ {os.path.basename(p)} — {len(dets)} لوحة")
    _upload_state.update(running=False, added=n)
    _upload_state["log"].append(f"=== أُضيف {n} لوحة جديدة للتوسيم ===")


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("images")
    saved = []
    for f in files:
        if not f.filename:
            continue
        dest = os.path.join(UPLOAD_DIR, os.path.basename(f.filename))
        f.save(dest)
        saved.append(dest)
    if not saved:
        return jsonify({"ok": False, "msg": "no files"})
    threading.Thread(target=_process_uploads, args=(saved,), daemon=True).start()
    return jsonify({"ok": True, "count": len(saved)})


@app.route("/api/upload_status")
def api_upload_status():
    return jsonify(_upload_state)


# ---- routes: training ------------------------------------------------------
_train = {"running": False, "log": [], "val_acc": None, "best_acc": None}


def _append(line):
    _train["log"].append(line)
    m = re.search(r"val_acc:\s*([0-9.eE+-]+)", line)
    if m:
        try:
            _train["val_acc"] = float(m.group(1))
        except ValueError:
            pass
    m2 = re.search(r"val_acc improved from [0-9.]+ to ([0-9.]+)", line)
    if m2:
        _train["best_acc"] = float(m2.group(1))


def _run_training():
    _train.update(running=True, log=[], val_acc=None, best_acc=None)
    # 1) build dataset from current labels
    _append("$ بناء مجموعة البيانات من التسميات…")
    p = subprocess.run([PY, os.path.join(SCRIPTS, "3_build_dataset.py"),
                        "--real-weight", "12", "--val-frac", "0.15"],
                       cwd=ROOT, capture_output=True, text=True)
    for ln in (p.stdout + p.stderr).splitlines():
        _append(ln)

    # 2) fine-tune
    _append("$ بدء fine-tuning من الموديل العالمي…")
    env = dict(os.environ, KERAS_BACKEND="tensorflow")
    proc = subprocess.Popen(["bash", os.path.join(SCRIPTS, "4_train.sh")],
                            cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        _append(line.rstrip())
    proc.wait()

    # 3) export best model -> models/sudan_ocr.onnx so the app uses it
    _append("$ تصدير أفضل موديل…")
    out_dirs = sorted([d for d in os.listdir(os.path.join(ROOT, "training", "output"))
                       if os.path.isdir(os.path.join(ROOT, "training", "output", d))])
    if out_dirs:
        best = os.path.join(ROOT, "training", "output", out_dirs[-1], "best.keras")
        if os.path.exists(best):
            subprocess.run([FPO, "export", "--model", best,
                            "--plate-config-file",
                            os.path.join(ROOT, "training", "config", "sudan_plate.yaml"),
                            "--format", "onnx"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
            onnx = os.path.join(ROOT, "training", "output", out_dirs[-1], "best.onnx")
            if os.path.exists(onnx):
                import shutil
                shutil.copy(onnx, os.path.join(MODELS, "sudan_ocr.onnx"))
                shutil.copy(os.path.join(ROOT, "training", "config", "sudan_plate.yaml"),
                            os.path.join(MODELS, "sudan_plate.yaml"))
                _append("✓ حُفظ الموديل في models/sudan_ocr.onnx")

    if _train["best_acc"] is not None:
        _append(f"=== انتهى التدريب. أفضل دقة: {_train['best_acc']*100:.0f}% ===")
    else:
        _append("=== انتهى التدريب ===")
    _train["running"] = False


@app.route("/api/train", methods=["POST"])
def api_train():
    if _train["running"]:
        return jsonify({"ok": False, "msg": "running"})
    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/train_status")
def api_train_status():
    return jsonify({"running": _train["running"], "log": _train["log"][-300:],
                    "val_acc": _train["val_acc"], "best_acc": _train["best_acc"]})


PAGE = r"""<!doctype html>
<html lang="ar" dir="rtl"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>استوديو لوحات السودان</title>
<style>
 :root{font-family:-apple-system,Segoe UI,Roboto,sans-serif}
 body{margin:0;background:#0f1115;color:#e8eaed}
 header{display:flex;gap:10px;align-items:center;padding:12px 18px;background:#161a22;
        border-bottom:1px solid #262b36;position:sticky;top:0;z-index:5}
 header h1{font-size:17px;margin:0;flex:1}
 .tab{padding:8px 16px;border-radius:8px;background:#222834;cursor:pointer;border:1px solid #2c3340}
 .tab.active{background:#2563eb;border-color:#2563eb}
 .wrap{max-width:780px;margin:22px auto;padding:0 16px}
 .card{background:#161a22;border:1px solid #262b36;border-radius:14px;padding:20px;text-align:center}
 .plate{max-width:100%;max-height:300px;border-radius:10px;background:#fff;padding:6px}
 .guess{color:#9aa4b2;margin:10px 0 4px;font-size:14px}
 input[type=text]{font-size:28px;text-align:center;letter-spacing:3px;padding:12px;width:88%;
   border-radius:10px;border:2px solid #2c3340;background:#0f1115;color:#fff;text-transform:uppercase}
 input[type=text]:focus{border-color:#2563eb;outline:none}
 .row{display:flex;gap:10px;justify-content:center;margin-top:16px;flex-wrap:wrap}
 button{font-size:16px;padding:11px 20px;border-radius:10px;border:0;cursor:pointer;color:#fff}
 .save{background:#16a34a}.skip{background:#64748b}.prev{background:#334155}.go{background:#2563eb}
 .bar{height:10px;background:#222834;border-radius:6px;overflow:hidden;margin:14px 0}
 .bar>i{display:block;height:100%;background:#2563eb;width:0;transition:width .3s}
 .meta{color:#9aa4b2;font-size:13px;margin-top:8px}
 #log,#ulog{background:#0a0c10;border:1px solid #262b36;border-radius:10px;padding:12px;height:46vh;
   overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;white-space:pre-wrap;text-align:left;direction:ltr}
 .hide{display:none}
 kbd{background:#222834;border:1px solid #2c3340;border-radius:5px;padding:1px 6px;font-size:12px}
 code{background:#222834;border-radius:5px;padding:2px 8px;color:#7dd3fc;font-size:15px}
 .help{background:#1a2030;border:1px solid #2c3a52;border-radius:12px;padding:14px 18px;
   margin-bottom:16px;font-size:14px;line-height:2;color:#cbd5e1;text-align:right}
 .acc{font-size:64px;font-weight:800;margin:6px}
 .acc.good{color:#22c55e}.acc.mid{color:#eab308}.acc.low{color:#ef4444}
 .drop{border:2px dashed #2c3a52;border-radius:14px;padding:36px;color:#9aa4b2;cursor:pointer}
 .drop.over{border-color:#2563eb;background:#10203a}
 .gtbar{height:14px;background:#222834;border-radius:8px;overflow:hidden;margin:10px 0}
 .gtbar>i{display:block;height:100%;background:linear-gradient(90deg,#ef4444,#eab308,#22c55e)}
</style></head>
<body>
<header>
 <h1>🇸🇩 استوديو تدريب لوحات السودان</h1>
 <div class="tab active" id="t_up" onclick="show('up')">رفع الصور</div>
 <div class="tab" id="t_lbl" onclick="show('lbl')">التوسيم</div>
 <div class="tab" id="t_tr" onclick="show('tr')">التدريب</div>
</header>

<!-- UPLOAD -->
<div class="wrap" id="v_up">
 <div class="card">
  <div class="help">ارفع صور سيارات (متعددة). سيتم كشف اللوحات وقصّها تلقائياً وإضافتها لمجموعة التوسيم.</div>
  <div class="drop" id="drop" onclick="document.getElementById('file').click()">
   📤 اسحب الصور هنا أو اضغط للاختيار
  </div>
  <input type="file" id="file" multiple accept="image/*" class="hide" onchange="upload(this.files)">
  <div id="ulog" style="margin-top:14px">—</div>
 </div>
</div>

<!-- LABEL -->
<div class="wrap hide" id="v_lbl">
 <div class="help"><b>اكتب السطر اللاتيني كاملاً بدون مسافات</b> — مثل <code>7KH10346</code>.
  الاقتراح صحيح؟ <kbd>Enter</kbd>. به خطأ؟ اكتب اللوحة كاملة. غير واضح؟ <b>تخطّي</b>.</div>
 <div class="card">
  <div class="bar"><i id="prog"></i></div>
  <div class="meta" id="counter">…</div>
  <img id="plate" class="plate" src="">
  <div class="guess">اقتراح: <b id="guess">…</b></div>
  <input id="txt" type="text" autocomplete="off" placeholder="7KH10346">
  <div class="row">
   <button class="prev" onclick="prev()">◀ السابق</button>
   <button class="skip" onclick="skip()">تخطّي</button>
   <button class="save" onclick="saveLbl()">حفظ والتالي ▶</button>
  </div>
 </div>
</div>

<!-- TRAIN -->
<div class="wrap hide" id="v_tr">
 <div class="card">
  <div class="meta">دقة الموديل (على لوحات محجوزة لم يرها)</div>
  <div class="acc low" id="accval">—</div>
  <div class="gtbar"><i id="accbar" style="width:0"></i></div>
  <div class="meta" id="trmeta">الهدف: 100% — كل ما رفعت ووسّمت أكثر، ارتفعت الدقة.</div>
  <div class="row"><button class="go" id="btnTrain" onclick="startTrain()">🚀 درّب الآن</button></div>
  <div id="log" style="margin-top:14px">—</div>
 </div>
</div>

<script>
let plates=[], i=0;
function show(w){
 for(const [k,v] of Object.entries({up:'v_up',lbl:'v_lbl',tr:'v_tr'}))
   document.getElementById(v).classList.toggle('hide',k!==w);
 for(const [k,v] of Object.entries({up:'t_up',lbl:'t_lbl',tr:'t_tr'}))
   document.getElementById(v).classList.toggle('active',k===w);
 if(w==='lbl')boot();
}
// ---- upload ----
const drop=document.getElementById('drop');
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>upload(ev.dataTransfer.files));
async function upload(files){
 if(!files.length)return;
 const fd=new FormData();
 for(const f of files)fd.append('images',f);
 document.getElementById('ulog').textContent='جارٍ الرفع والمعالجة…';
 await fetch('/api/upload',{method:'POST',body:fd});
 pollUpload();
}
async function pollUpload(){
 const d=await (await fetch('/api/upload_status')).json();
 document.getElementById('ulog').textContent=d.log.join('\n');
 if(d.running)setTimeout(pollUpload,1200);
 else document.getElementById('ulog').textContent+='\n\n✓ جاهز — انتقل لتبويب التوسيم.';
}
// ---- label ----
async function boot(){
 const d=await (await fetch('/api/plates')).json();
 plates=d.plates; i=plates.findIndex(p=>!p.label); if(i<0)i=0; render();
}
async function render(){
 if(!plates.length){document.getElementById('counter').textContent='لا لوحات بعد — ارفع صوراً أولاً';return;}
 const p=plates[i];
 document.getElementById('plate').src='/api/image/'+encodeURIComponent(p.name)+'?t='+Date.now();
 const done=plates.filter(x=>x.label).length;
 document.getElementById('counter').textContent=`لوحة ${i+1}/${plates.length} — مُوسَّم: ${done}`;
 document.getElementById('prog').style.width=(100*done/plates.length)+'%';
 const txt=document.getElementById('txt'); txt.value=p.label||''; txt.focus();
 document.getElementById('guess').textContent='…';
 const g=await (await fetch('/api/guess/'+encodeURIComponent(p.name))).json();
 document.getElementById('guess').textContent=g.guess||'(لا شيء)';
 if(!txt.value){txt.value=g.guess||''; txt.select();}
}
async function saveLbl(){
 const p=plates[i], text=document.getElementById('txt').value.trim().toUpperCase();
 await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:p.name,text})});
 p.label=text; nextLbl();
}
function nextLbl(){ if(i<plates.length-1){i++;render()} else alert('انتهيت! 🎉 انتقل للتدريب'); }
function prev(){ if(i>0){i--;render()} }
function skip(){ nextLbl(); }
document.addEventListener('keydown',e=>{
 if(!document.getElementById('v_lbl').classList.contains('hide') && e.key==='Enter'){e.preventDefault();saveLbl();}
});
// ---- train ----
function setAcc(a){
 const el=document.getElementById('accval'), bar=document.getElementById('accbar');
 if(a==null){el.textContent='—';return;}
 const pct=Math.round(a*100); el.textContent=pct+'%';
 el.className='acc '+(pct>=90?'good':pct>=70?'mid':'low');
 bar.style.width=pct+'%';
}
async function startTrain(){
 document.getElementById('btnTrain').disabled=true;
 document.getElementById('btnTrain').textContent='⏳ يتدرّب…';
 await fetch('/api/train',{method:'POST'}); pollTrain();
}
async function pollTrain(){
 const d=await (await fetch('/api/train_status')).json();
 document.getElementById('log').textContent=d.log.join('\n');
 document.getElementById('log').scrollTop=1e9;
 setAcc(d.best_acc!=null?d.best_acc:d.val_acc);
 if(d.running)setTimeout(pollTrain,1500);
 else{
   document.getElementById('btnTrain').disabled=false;
   document.getElementById('btnTrain').textContent='🚀 درّب الآن';
   setAcc(d.best_acc);
 }
}
// show current accuracy on load
fetch('/api/train_status').then(r=>r.json()).then(d=>{if(d.best_acc!=null)setAcc(d.best_acc)});
</script>
</body></html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
