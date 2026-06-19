# Training the OCR model on real Sudanese plates

A pipeline for fine-tuning a plate-reading model for Sudanese plates, using
**real photos only** — no synthetic data.

## The honest truth before you start

A reliable OCR model needs **at least ~1000 labeled real plates**. With fewer
(a few dozen), the model just *memorizes* the images and can't read anything
new. So this pipeline is built for **incremental growth**: every time you add
more photos to `../input/`, your set gets bigger until there's enough to train
on properly.

## The steps

```bash
cd training/scripts

# 1) Crop the plates out of the car photos (uses the YOLO detector)
python 1_crop_plates.py ../../input

# 2) Label the plates (semi-automatic: it suggests a reading, you just correct it)
python 2_make_labels.py

# 3) Build the train/val files from the labeled plates
python 3_build_dataset.py --real-weight 8

# 4) Train the model
bash 4_train.sh
```

## Where things stand

- There are currently **~120 labeled plates** in `dataset/labels.csv` — a solid
  start, and enough to fine-tune a working model (the shipped `sudan_ocr.onnx`
  was trained from this set).
- Keep adding Sudanese plate photos to `../input/`, then re-run steps 1–2 to
  grow the labeled set.
- The more labeled plates you reach (aim for **500–1000+**), the more robust the
  model gets. Re-run steps 3–4 to retrain.

## Why not use Egyptian / Saudi plates?

We looked into it and tested it: the Egyptian plate is **fundamentally
different** (no Latin line, different layout and characters), and training on it
can actually **hurt** Sudanese accuracy rather than help. There's no substitute
for real Sudanese plates.
