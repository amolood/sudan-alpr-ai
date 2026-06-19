<h1 align="center">🇸🇩 Sudan ALPR</h1>

<p align="center">
  <strong>Automatic license-plate recognition for Sudanese plates.</strong><br>
  Finds the plate, reads the serial, confirms it's Sudanese, and decodes the state (wilaya) — all locally.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Accuracy" src="https://img.shields.io/badge/OCR%20exact--match-82.6%25-success">
  <img alt="Country" src="https://img.shields.io/badge/country%20detection-100%25-success">
  <img alt="Tests" src="https://img.shields.io/badge/tests-42%20passing-brightgreen">
  <img alt="Built with" src="https://img.shields.io/badge/built%20with-FastALPR%20%2B%20ONNX-orange">
</p>

<p align="center">
  <a href="https://github.com/amolood/sudan-alpr-ai/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/amolood/sudan-alpr-ai?style=social"></a>
  <a href="https://github.com/amolood/sudan-alpr-ai/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/amolood/sudan-alpr-ai?style=social"></a>
  <a href="https://github.com/amolood/sudan-alpr-ai/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/amolood/sudan-alpr-ai"></a>
  <a href="https://github.com/amolood/sudan-alpr-ai/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/amolood/sudan-alpr-ai"></a>
  <a href="https://github.com/amolood/sudan-alpr-ai/issues"><img alt="Issues" src="https://img.shields.io/github/issues/amolood/sudan-alpr-ai"></a>
</p>

<p align="center">
  <em>⭐ If this is useful to you, a star helps others find it.</em>
</p>

<p align="center">
  <img src="docs/demo_plate.png" alt="A real Sudanese plate detected and read as 2G479 (Gezira)" width="520">
  <br>
  <em>A real plate, detected and read: <code>2G479</code> → Sudan / Gezira</em>
</p>

It runs two deep-learning models locally on top of
[FastALPR](https://github.com/ankandrew/fast-alpr) — no cloud, no API keys. I
wrote it because the old template-matching version I had fell apart the moment a
photo was taken at an angle or from a distance, which is basically every real
photo. This one holds up on messy, real-world shots.

---

## Contents

- [How it works](#how-it-works) · [Requirements](#requirements) · [Install](#install)
- [Project layout](#project-layout) · [Running it](#running-it) · [Output](#output)
- [Country, class & state recognition](#country-class-and-state-recognition)
- [Accuracy & benchmark](#benchmark) · [Tests](#tests)
- [Honest caveats](#honest-caveats) · [Training](training/README.md)
- [Contributors](#contributors) · [License](#license)

---

## At a glance

| | |
|---|---|
| **What it reads** | Sudanese plates: serial, country, plate class, state |
| **Plate classes** | 16 (private, government, police, army, diplomatic, UN, NGO, investment, transit, temporary…) |
| **OCR exact-match** | **82.6%** (vs 0% for the off-the-shelf model) |
| **Country detection** | **100%** on the labeled set |
| **Runs on** | CPU or Apple Silicon (CoreML), fully offline after first run |
| **Stack** | YOLO-v9 detector + fine-tuned CCT transformer OCR, via ONNX Runtime |

## How it works

It's a two-stage pipeline, same idea every serious ANPR system uses:

```
car photo
   │
   ▼
[1] detect   →  a YOLO-v9 model locates the plate and crops it out
   │
   ▼
[2] read     →  a CCT transformer OCR model turns the crop into text → 3KH3476
   │
   ▼
[3] interpret →  confirm it's Sudanese + decode the state → Sudan / Khartoum
```

Both models run on your machine through ONNX Runtime. On Apple Silicon they use
CoreML automatically, so it's quick. The OCR model is fine-tuned on real
Sudanese plates, so it reads the Latin serial line (e.g. `3KH 3476`) even when
the lighting is bad or the plate is tilted.

## Requirements

- **Python 3.11 or 3.12.** Not 3.14 — onnxruntime doesn't ship a wheel for it
  yet, and you'll just hit an install error.
- Internet on the first run only, to pull the model weights (~11 MB). After
  that it works offline.

## Install

A ready-to-go `venv/` is included. If you'd rather build it yourself:

```bash
cd sudan-alpr-ai
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Project layout

```
sudan-alpr-ai/
├── recognize.py            reader: global OCR + manual column splitting
├── recognize_trained.py    reader: the fine-tuned model (most accurate)
├── recognize_video.py      reader: video files / live camera, frame by frame
├── sudan_plate.py          interpreter: is-it-Sudanese? + class + state
├── benchmark.py            measures accuracy and compares the models
├── make_chart.py           renders the benchmark chart in docs/
├── test_sudan_plate.py     unit tests for the interpreter (pytest)
├── webapp.py               drag-and-drop web demo (Flask)
├── requirements.txt
├── models/
│   ├── sudan_ocr.onnx      OCR model fine-tuned on Sudanese plates
│   └── sudan_plate.yaml    model config (alphabet, input size, …)
├── input/                  drop your car photos here
├── output/                 annotated images + results.json + benchmark.json
└── training/               pipeline for fine-tuning the OCR on real plates
    ├── dataset/
    │   ├── good_plates/    cropped plates (used by training and benchmarks)
    │   ├── labels.csv      hand-verified ground truth
    │   ├── train.csv       training split
    │   └── val.csv         held-out validation split
    └── scripts/            crop → label → build dataset → train
```

The data flows in a loop: photos in `input/` get cropped by the detector into
`training/dataset/`, you label them in `labels.csv`, training produces
`models/sudan_ocr.onnx`, and `benchmark.py` scores that model back against
`labels.csv`. Collect more photos, run it again, get a better model.

Training details live in [`training/README.md`](training/README.md).

## Running it

There are two readers. Use `recognize_trained.py` unless you have a reason not
to — it's the accurate one.

| Script | Input | When to reach for it |
|---|---|---|
| `recognize_trained.py` | images | **recommended** — fine-tuned OCR, reads the Latin line directly |
| `recognize.py` | images | no trained model needed; leans on the known plate layout |
| `recognize_video.py` | video / camera | read plates from a clip or live webcam |
| `webapp.py` | browser | drag-and-drop web demo |

```bash
# the fine-tuned reader (recommended)
./venv/bin/python recognize_trained.py input/
./venv/bin/python recognize_trained.py input/some_car.jpg

# the column-split reader
./venv/bin/python recognize.py input/test_plate.png   # one image
./venv/bin/python recognize.py input/                 # a whole folder
./venv/bin/python recognize.py input/ --det-conf 0.5  # drop weak detections
```

Drop your photos in `input/`, point the script at it, done.

### Video & live camera

```bash
./venv/bin/python recognize_video.py clip.mp4            # a video file
./venv/bin/python recognize_video.py 0                   # the default webcam
./venv/bin/python recognize_video.py clip.mp4 --save out.mp4 --every 5
```

It runs detection on every Nth frame and de-duplicates, so a plate passing
through is reported once with a frame-count tally — not once per frame. Press
`q` in the preview window to stop.

### Web demo

```bash
./venv/bin/pip install flask     # one time
./venv/bin/python webapp.py      # then open http://127.0.0.1:5000
```

A single-page app: drop a car photo and see the boxed plate with its reading,
country, class, and state. Same pipeline as the CLI.

## Output

In the terminal you get the plate text plus country, class/state, and
confidence. Private plates show their state; special plates show their class:

```
📷 my_car.jpg
    🔖 3KH3476      🇸🇩 Sudan / Khartoum         (country 95% | detect 90%)
    🔖 POLICE00000  🇸🇩 Sudan / Police           (country 97% | detect 88%)
```

And in `output/`:

- `annotated_<image>` — the original photo with a box drawn around the plate
  and the reading (plus `KH/Sudan` or the class) written above it.
- `results.json` — everything in JSON: text, country, plate class, state,
  confidence scores, and the plate's bounding box.

## Country, class, and state recognition

Once the text is read, it passes through the Sudanese plate interpreter in
[`sudan_plate.py`](sudan_plate.py), which answers three things:

- **Is this a Sudanese plate?** Yes/no, with a confidence score.
- **What class of plate is it?** Private, government, police, army, diplomatic,
  UN, NGO, transit, temporary… (see below).
- **Which state (wilaya)?** For private plates, it decodes the state letters
  into a name, in both English and Arabic.

### Plate classes

Sudan doesn't issue one plate format — the General Directorate of Traffic uses
a whole family, told apart by colour and a text marker. Here's the official
reference board from the Directorate, which is what these classes are modeled
on:

<p align="center">
  <img src="docs/plate_types_reference.png" alt="Sudan General Directorate of Traffic — official plate types reference board" width="420">
  <br>
  <em>Official Sudanese plate types — General Directorate of Traffic</em>
</p>

The interpreter knows all of them:

Each class is matched by a Latin marker, an Arabic marker, or both — every
entry transcribed straight off the board, plate by plate:

| Class | Latin marker | Arabic | Colour |
|---|---|---|---|
| Private | `<digit><state><serial>` | خصوصي | silver / white |
| Government | `GOV` | حكومة | yellow |
| Armed Forces | `ARMY` | القوات المسلحة | red |
| Police | `POLICE` | الشرطة | white / blue |
| United Nations | `U.N` | الأمم المتحدة | red & blue variants |
| Diplomatic Missions | `C.D` | هيئات دبلوماسية | red |
| Consular Missions | `H.C` | هيئات قنصلية | green |
| NGO | `N.G.O` | منظمة غير حكومية | yellow |
| International Organizations | `I.O` | منظمات دولية | white |
| Sudanese Red Crescent | — | الهلال الأحمر السوداني | white |
| Limousine | — | ليموزين | silver |
| Investment / Commercial | — | استثمار | green / black |
| Transit | `TRANSIT` | عبور | silver |
| Temporary | — | مؤقتة | white |
| Temporary (Express) | — | مؤقتة سريع | white |
| Temporary (Domestic) | — | مؤقتة داخلي | white |

Several plates (Red Crescent, Limousine, Investment, Temporary) carry **only an
Arabic word** — no Latin code — so the interpreter matches Arabic markers too,
not just Latin ones. Investment plates also carry a state code (e.g. `KH9`,
`RS`), which still gets decoded to a wilaya; they're told apart from ordinary
state plates by their green/black colour and the word **استثمار**.

The **text marker decides the class** — so it still works on a greyscale or
badly-lit photo. If the caller also measures the plate's **colour** (the
column-split reader does, via `dominant_plate_color`), that colour is used as
*corroboration* and nudges the confidence up, but it's never the sole signal.

So a `POLICE 00000` plate reads as Police, an `ARMY 00000` as Armed Forces, a
`ليموزين` plate as Limousine — and a normal `7KH10346` stays Private with its
state decoded to Khartoum.

### Why structure, not guessing

A Sudanese civilian plate has a fixed, well-known layout:

```
registration digit ─┐   ┌─ state letters   ┌─ serial number
                    7      KH                10346    →  "7KH10346"
```

The off-the-shelf global model used to *guess* the country, and it guessed
wrong — Sudan isn't in its 65-country list, so it would return whatever random
country looked closest. So I don't ask a model. I check the pattern itself:
digit, then state letters, then digits. That shape is specifically Sudanese.
If the text matches, the plate is Sudanese, and the interpreter records *why* it
decided that — no hallucinated answers.

### State codes — and an honest caveat

Here's the part most ALPR write-ups quietly skip. Sudan Police publish the plate
letters **in Arabic only**. There is no official, published table mapping each
Arabic letter to a Latin code (`KH`, `NK`, …) or to a state name. So I only
claim what I can actually support:

| Latin code | State | Arabic letter | Evidence |
|---|---|---|---|
| `KH` | Khartoum | خ | **confirmed** (published source) |
| `G` | Gezira | ج | observed in data |
| `NS` | River Nile | ش | observed in data |
| `NK` | North Kordofan | ش ك | observed in data |
| `WK` | West Kordofan | غ ك | observed in data |
| `WN` | White Nile | و د | observed in data |
| `RS` | Red Sea | ب ح | observed in data |

Only **`KH` is confirmed** against a published source. The rest are codes that
genuinely appear in our own labeled data and photos — strong evidence they're
real — but I haven't found an official source that names them, so the
interpreter tags them *"observed, not officially confirmed"* rather than stating
them as fact.

Codes I previously **guessed** for the other states (South Kordofan, the Darfur
states, Kassala, Sennar…) have been **removed**. Their Arabic plate letters are
kept in the code (`UNMAPPED_ARABIC_LETTERS`) for reference, but with no Latin
code — inventing one I can't back up would defeat the purpose. If you have an
authoritative source for those mappings, they're a one-line addition each.

The decoder is also forgiving of bad OCR: if the reader drops the registration
digit (`KH5404`) or doubles it (`10KH6009`), the plate is still recognized as
Sudanese — just with lower confidence instead of being thrown out.

### How well does it do?

Run the country benchmark and see for yourself:

```bash
./venv/bin/python benchmark.py --country --models sudan
```

| Model | Flagged Sudanese | Accuracy | State correct | State accuracy |
|---|---|---|---|---|
| **sudan (fine-tuned)** | 121/121 | **100.0%** | 118/121 | **97.5%** |

Every single plate got correctly flagged as Sudanese, and 97.5% landed on the
right state. The three state misses weren't logic errors — they were truncated
labels (digits with no state letter, nothing to decode). The point is the
number is measured, not asserted.

## How I got the accuracy up

I tried two approaches. Both are benchmarked below.

### First approach: splitting the columns by hand (`recognize.py`)

The global model on its own *failed* on Sudanese plates. It tries to read the
whole plate in one shot, mixes up the big Arabic line with the small Latin one,
and produces garbage. The first fix exploited the known layout:

```
┌──────────────────────────┐
│   SUDAN       السودان     │  ← top third (ignored)
│  ٧ خ          ١٠٣٤٦       │  ← big Arabic line
│  7KH          10346       │  ← small Latin line ← we read this
└──────────────────────────┘
   letter column  number column
```

1. **Crop each column separately** and read it on its own — letters (`7KH`) on
   the left, digits (`10346`) on the right.
2. **Upscale each crop 4×** before reading, since the digits are tiny.
3. **Fix common confusions** in the number column (`O→0`, `L→1`, `S→5`…),
   because it's digits only.
4. **Fall back** to reading the whole plate if the split fails on a tiny or
   blurry image.

On the first five real plates I tested by hand, this took accuracy from
**0/5 to 4/5**:

| Plate | Read | |
|---|---|---|
| `1KH 5490` | `1KH5490` | ✅ |
| `1NS 180` | `1NS180` | ✅ |
| `2KH 14514` | `2KH14514` | ✅ |
| `7KH 10346` | `7KH10346` | ✅ |
| 76×39 px plate (way too far) | wrong | ❌ too small |

### Second approach: training a custom model (`recognize_trained.py`)

Instead of hand-written rules, I took the global model and fine-tuned it on
labeled real Sudanese plates. The result reads the Latin line directly, no
rules needed, and hits **~83% exact-match** versus **0%** for the global model
on the same images. Full numbers below; training steps are in
[`training/README.md`](training/README.md).

## Benchmark

"It works great" means nothing without a number behind it. So there's a
benchmark script that scores the models objectively, on the **same images**,
against **hand-labeled ground truth** (`training/dataset/labels.csv`).

### How the benchmark works

1. It takes the **pre-cropped** plates from `training/dataset/good_plates/`.
2. It runs each one through the OCR model **only** — the detector is out of the
   loop — so the numbers reflect *reading* accuracy, not detection, and the
   comparison is fair (identical inputs for every model).
3. It compares the model's text to the correct answer, character by character
   (after normalizing both to letters and digits).

### The metrics

| Metric | What it means | Better |
|---|---|---|
| **Exact-match** | the whole plate is read correctly, character for character | higher |
| **CER** (character error rate) | edit distance ÷ characters; catches "almost right" | lower |
| **ms/plate**, **plates/s** | speed | faster |

### What's being compared

- **global** — the stock `cct-xs-v2-global-model` (65+ countries). Never saw a
  Sudanese plate. This is the baseline.
- **sudan** — my fine-tuned `models/sudan_ocr.onnx`, trained on real Sudanese
  plates.

### Commands

```bash
# every labeled plate, both models
./venv/bin/python benchmark.py

# the held-out validation set only (the model never trained on these)
./venv/bin/python benchmark.py --split val

# one model, and print every misread
./venv/bin/python benchmark.py --models sudan --show-errors

# country (is-it-Sudanese?) + state recognition
./venv/bin/python benchmark.py --country --models sudan

# dump the results to JSON
./venv/bin/python benchmark.py --json output/benchmark.json
```

### The numbers

<p align="center">
  <img src="docs/benchmark.png" alt="Benchmark chart: exact-match accuracy and character error rate, global vs fine-tuned" width="720">
</p>

Across **all 121 labeled plates:**

| Model | Exact-match | Accuracy | CER | ms/plate | plates/s |
|---|---|---|---|---|---|
| global (baseline) | 0/121 | **0.0%** | 78.3% | ~7.1 | ~140 |
| **sudan (fine-tuned)** | 100/121 | **82.6%** | **7.9%** | ~6.0 | ~167 |

The chart above is generated from the benchmark output — run
`./venv/bin/python make_chart.py` to regenerate it from
`output/benchmark.json`.

On the **held-out validation set (19 plates the model never saw):**

| Model | Exact-match | Accuracy | CER |
|---|---|---|---|
| **sudan (fine-tuned)** | 16/19 | **84.2%** | **4.5%** |

The global model scores a flat 0% because it reads the whole plate at once,
blends the Arabic and Latin lines, and spits out junk like `AAA` or `41SA1`.
The fine-tuned model jumps to ~83% on the exact same crops. That gap is the
whole argument for training it.

These numbers are reproducible — run the command and you'll get the same thing
(ms/plate will wobble a bit depending on your machine).

### Reading the results

- The plates that fail are usually **too small or blurry**, or lost a character
  when they were cropped (e.g. `1K91490` read as `1KH91490` — one extra
  character). There's nothing in the pixels to read. Use `--show-errors` to see
  exactly what failed and why.
- A **low CER with a lower exact-match** means the model is *close* — usually
  one character off, not lost. That's the healthy kind of wrong: more training
  data closes it.

## Tests

The interpreter (`sudan_plate.py`) is covered by a unit-test suite that locks in
every behaviour: country detection, all 16 plate classes (Latin *and* Arabic
markers), state decoding, the collision guards (e.g. `1CDR500` must stay private,
not diplomatic), Latin folding (`TRANŞIT`), OCR tolerance, and the honesty rules
around unconfirmed state codes.

```bash
./venv/bin/python -m pytest -q          # 42 tests, runs in well under a second
```

The interpreter is pure Python, so the tests need nothing but `pytest` — no
models, no GPU. They run in a fraction of a second, so it's easy to run them
before every change.

## Honest caveats

1. **Failures are almost always image quality**, not the model. A 76×39 px
   plate (tiny and blurry) or a character clipped during cropping — no model
   can read detail that isn't in the photo. Closer, sharper shots fix it. Run
   `benchmark.py --show-errors` for the full list of what fails and why.
2. **The country is no longer guessed by the global model** (which used to
   return wrong countries because Sudan isn't in its 65-country list). It's now
   decided from the plate's structure in [`sudan_plate.py`](sudan_plate.py),
   at **100%** on this dataset — see the country/state section above.
3. **The Arabic line** (`٣خ ٣٤٧٦`): the OCR is tuned for Latin text, so we read
   the bottom Latin line, which carries the same information. If you specifically
   need the Arabic text read accurately, that means fine-tuning an OCR on real
   Sudanese plates — doable, but it needs a labeled dataset.
4. **To push accuracy even higher** on Sudanese plates specifically: gather
   200–500 plate photos, annotate them, and fine-tune both the detector and the
   OCR on them. The training pipeline here is set up for exactly that.

## Versus the old version

| | Old version (MATLAB / corr2) | This version (AI) |
|---|---|---|
| Method | pixel template matching | deep learning (YOLO + transformer OCR) |
| Angled real photos | usually failed | handles them fine |
| Setup | needs MATLAB | free Python |
| Accuracy | limited | high |

---

## Contributors

Thanks to everyone who has helped improve this project 💚

<a href="https://github.com/amolood/sudan-alpr-ai/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=amolood/sudan-alpr-ai" />
</a>

| Contributor | Role |
|---|---|
| [@amolood](https://github.com/amolood) | Author & maintainer — pipeline, interpreter, benchmark, training |
| [@mhamidawad](https://github.com/mhamidawad) | Training script hardening — WandB tracking, checkpoint resume, pre-flight checks |

Contributions are welcome — open an issue or a pull request. Good first areas:
adding verified state codes, gathering labeled plates for training, or reading
the Arabic line directly.

## Acknowledgements

- [FastALPR](https://github.com/ankandrew/fast-alpr) — the detector + OCR engine this builds on.
- The official Sudan General Directorate of Traffic plate-types board, used to verify every plate class.

## License

Released under the [MIT License](LICENSE) — free to use, learn from, and build on.

<p align="center">
  <sub>Built for educational purposes · 🇸🇩</sub>
</p>
