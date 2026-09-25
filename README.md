# ANPR Project — Code & Reports

Vehicle orientation classification and dataset engineering for an ANPR pipeline.
All scripts written during the internship, organised by pipeline stage.

**Server location:** `/mnt/datadisk/avani/front_back/main/`
**Virtualenv:** `source /mnt/datadisk/avani/.venv/bin/activate`
**YOLOv5 repo:** `/mnt/datadisk/avani/yolov5`

---

## Folder structure

### `1_pipeline/` — data acquisition and preprocessing

| File | Purpose |
|---|---|
| `config.py` | Central configuration — paths, thresholds, ANPR endpoint, thread counts |
| `utils.py` | BK-tree, SHA256/pHash, ANPR client, checkpointing, thread-safe logging |
| `preprocess.py` | Deduplication + ANPR validation, resumable |
| `download_images.py` (+`_2`, `_3`) | Threaded, resumable, atomic bulk downloader (one per batch) |
| `merge_datasets.py` | Multi-source merge — images symlinked, labels copied |
| `add_dataset.py` | End-to-end orchestrator: extract URLs → download → label → merge → plate-label |

### `2_filtering_labelling/` — quality filtering and annotation

| File | Purpose |
|---|---|
| `make_yolo_labels.py` | Automated vehicle labelling with plate-visibility gating |
| `filter_brightness.py` | Brightness banding + visual sampling, then filter (threshold 40 used) |
| `filter_dataset.py` | Triage into clean / no_vehicle / no_plate |
| `add_plate_labels.py` | Idempotent `reg_plate` class augmentation via the ANPR service |
| `recover_plates_local.py` | Second-chance plate detection with a local model on ANPR failures |
| `split_car_other.py` | 5-class → 7-class relabel; crops non_bike boxes and asks COCO car vs other |

### `3_dataset_split/` — splitting and training

| File | Purpose |
|---|---|
| `split_dataset.py` | Original 70/20/10 split (4-class) |
| `split_merged.py` | 5-class split of the merged dataset |
| `split_dataset3.py` | 75/10/15 seeded-shuffle split — **use this one** |
| `train_yolov5.py` | Training launcher wrapper |

### `4_evaluation/` — metrics and error analysis

| File | Purpose |
|---|---|
| `eval_classification.py` | Pure classification accuracy + confusion matrix (ignores box quality) |
| `eval_metrics.py` | Precision/recall/F1/AP on arbitrary image+label folder pairs |
| `review_sample.py` | Score a sample, split into good/ and bad/, supports `--gt-class-map` |
| `find_bad_predictions.py` | Worst-prediction ranking with GT vs prediction overlays |
| `find_plate_errors.py` | Plate-specific failures: missed / spurious / loose_box / wrong_class |
| `class_distribution.py` | Per-class box and image counts |

### `5_visualisation/` — visual inspection

| File | Purpose |
|---|---|
| `infer_draw.py` | Draw **model predictions** on a random sample |
| `draw_labels.py` | Draw **ground-truth labels** on images (verify relabelling) |

### `reports/`

| File | Purpose |
|---|---|
| `report.md` / `.html` / `.docx` / `.pdf` | Internship engineering report, four formats |
| `ANPR_Project_Interview_Guide.md` | Project + deep learning interview prep |
| `build_docx.js` | Regenerates `report.docx` (needs `npm install docx`) |

---

## Standard workflow for a new data batch

```bash
# 1. Download + label + merge (one command)
python add_dataset.py --csv new_batch.csv --name batch4 --orientation back

# 2. Quality filter
python filter_brightness.py --images batch4_images --apply --threshold 40
python filter_dataset.py --images batch4_images --plate-conf 0.45

# 3. Plate labels
python add_plate_labels.py --images <merged>/images --labels <merged>/labels --workers 12

# 4. Split
python split_dataset3.py --base <merged>

# 5. Train (fine-tune from existing weights)
cd /mnt/datadisk/avani/yolov5
python train.py --weights <base.pt> --data <merged>/yolo_split/data.yaml \
  --hyp hyp.yaml --img 320 --batch-size 256 --epochs 300 \
  --device 0 --patience 50 --project <out> --name <run>
```

---

## Class schema

| ID | 4-class (v1) | 5-class (v2) | 7-class (v3) |
|---|---|---|---|
| 0 | bike_front | bike_front | bike_front |
| 1 | bike_back | bike_back | bike_back |
| 2 | non_bike_front | non_bike_front | car_front |
| 3 | non_bike_back | non_bike_back | car_back |
| 4 | — | reg_plate | other_front |
| 5 | — | — | other_back |
| 6 | — | — | reg_plate |

---

## Gotchas worth remembering

- **Label cache.** YOLOv5 caches labels by *directory path*, not content. After changing any labels, delete `labels.cache` next to them or training silently uses the old ones.
- **Model loading.** These `.pt` files are classic YOLOv5, not the `ultralytics` pip package. Load with `torch.hub.load(repo_dir, 'custom', path=weights, source='local')`.
- **Shuffle before splitting.** Merged datasets are in source order; splitting without a shuffle puts whole sources into single splits and produces misleading metrics.
- **Class-count changes.** Going from `nc=N` to `nc=M` reinitialises the detection head automatically — watch the `Transferred X/Y items` line to confirm the backbone carried over.
- **`--batch-size`**, not `--batch`, in `train.py`.
- **Line continuations.** Multi-line commands with trailing `\` get mangled on paste; use single-line `&&` chains.

---

## Note on file versions

These are the copies from the Cowork session. A few have drifted from the server:

- `infer_draw.py` — this copy has `--batch`; the server copy may not
- `split_dataset3.py` — this copy resolves `--base` to an absolute path (fixes the relative-path `data.yaml` bug)
- `review_sample.py` — this copy has `--gt-class-map`, matching the patch applied on the server

Push to the server with:

```bash
scp -r ~/Desktop/ANPR_Project/*/*.py root@e2e-102-18.ssdcloudindia.net:/mnt/datadisk/avani/front_back/main/
```
