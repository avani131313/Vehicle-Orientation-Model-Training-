# ANPR + Vehicle Classification — Project & Deep Learning Interview Guide

---

# PART 1 — YOUR PROJECT

## 1.1 The one-paragraph version (memorise this)

> "I built an end-to-end computer vision pipeline for automatic number plate recognition and vehicle classification, working with roughly 460,000 parking-transaction images. I owned the whole thing — data collection from transaction CSVs, deduplication, automated labelling, quality filtering, training, and evaluation. The final model is a single YOLOv5 detector that outputs five classes: vehicle type crossed with orientation (bike/non-bike × front/back) plus the registration plate itself. It hits 98% precision, 97% recall, and 93.8% mAP@0.5:0.95 on a held-out validation set."

That's your opener. Everything below is what you say when they dig in.

---

## 1.2 The problem

A parking system photographs every vehicle entering and exiting. For each image the system needs to know:

1. **Is there a vehicle?**
2. **What kind** — two-wheeler or four-wheeler? (different billing rates)
3. **Which way is it facing** — front or back? (determines where the plate is, and whether the image is an entry or exit shot)
4. **Where is the number plate?** (feed the crop to OCR for the actual registration number)

Doing 1–4 as separate models means four forward passes per image. Doing it as one detector with five classes means **one** pass. That's the core design decision of the project.

### The 5 classes

| ID | Class | Meaning |
|----|-------|---------|
| 0 | `bike_front` | Two-wheeler, front-facing |
| 1 | `bike_back` | Two-wheeler, rear-facing |
| 2 | `non_bike_front` | Car/truck/bus, front-facing |
| 3 | `non_bike_back` | Car/truck/bus, rear-facing |
| 4 | `reg_plate` | The number plate region |

**Why encode orientation into the class instead of a separate head?** Because YOLO's classification head is essentially free — it's already predicting a class per box. Adding orientation as a class multiplication (2 vehicle types × 2 orientations = 4) costs nothing extra at inference and avoids a second model. The trade-off is you need enough examples of every combination, which we had.

---

## 1.3 The data pipeline (this is where most of the work was)

Interviewers love this part because it shows you understand that **ML is 80% data work**. Be ready to talk about it for five minutes.

### Stage 1 — Acquisition
Image URLs were embedded in parking-transaction CSV exports. I wrote extractors to pull the URLs out and a **threaded, resumable downloader** (32 workers, atomic `.part` → rename on success, failure log, checkpointed) that pulled images in bulk. Three batches so far: ~86k, ~202k, ~274k images.

**Key detail to mention:** I verified zero overlap between batches using Python set intersection on the URL lists before downloading, so I didn't waste bandwidth or pollute the dataset with duplicates across batches.

### Stage 2 — Deduplication
Two levels:

- **Exact duplicates** — SHA256 hash of file bytes. O(n) with a hash set.
- **Near-duplicates** — perceptual hash (pHash) + **BK-tree**.

**The BK-tree is your best "I thought about scale" story.** Naive near-duplicate detection compares every image to every other image: O(n²). At 400,000 images that's 80 billion comparisons — completely infeasible. A BK-tree exploits the fact that Hamming distance is a **metric** (obeys the triangle inequality), which lets you prune entire subtrees during search. Lookup drops to roughly O(log n) in practice.

> **If they ask "how does a BK-tree work?"**: You insert items into a tree where each edge is labelled with the distance between parent and child. When searching for everything within distance `d` of a query, the triangle inequality tells you a child at edge-distance `k` can only contain matches if `|k − dist(query, node)| ≤ d`. Every child failing that test gets pruned along with its whole subtree.

### Stage 3 — Automated labelling
Hand-labelling 460k images is impossible. So:

- A **COCO-pretrained detector** (YOLO26x) finds vehicles and gives a coarse type — COCO's `motorcycle`/`bicycle` classes map to "bike", `car`/`truck`/`bus` map to "non_bike".
- Orientation came from the **batch context** — some folders were known to be all-rear or all-front camera angles, so orientation was assigned per-batch rather than predicted.
- Plate boxes came from an **existing ANPR microservice** (HTTP endpoint returning plate bbox + detection confidence + OCR text + OCR confidence). I converted its pixel bboxes to normalised YOLO `cx cy w h` format and appended them as class 4.

**Be honest about this in the interview** — it's a strength, not a weakness. Say: *"I used weak supervision — an ensemble of a pretrained detector and an existing production ANPR service — to generate labels at a scale hand-annotation couldn't reach. I then validated label quality by sampling and visually inspecting overlays."*

### Stage 4 — Quality filtering
This is what separates a working model from a broken one. Three filters:

**a) Brightness filter.** Many night images were so dark that only headlights and the plate were visible — front/back is genuinely undecidable, even for a human. I scored mean brightness on a 1/8-scale grayscale read (`cv2.IMREAD_REDUCED_GRAYSCALE_8` — fast and memory-light at scale), banded the distribution, **visually reviewed samples from each band**, and cut at a mean-brightness threshold of 40.

> **Why this matters in interview:** it shows you don't just pick thresholds arbitrarily — you looked at the data, sampled each band, and made an informed cut. Say that explicitly.

**b) No-vehicle filter.** Images where the detector found no vehicle box above confidence + minimum-area thresholds (min 2% of image area — filters out tiny spurious detections in the background).

**c) No-plate filter.** Images where the ANPR service found no plate above a confidence threshold. If the plate isn't visible, the image is useless for the downstream OCR task, so it's training noise.

Rejected images were **moved, never deleted** — every filtering decision was reversible and logged to CSV. This saved me more than once.

**d) Second-chance recovery.** For ~10,600 images where the ANPR service found no confident plate, I ran a *different* local plate-detection model as a second opinion. Recovered images got their plate box plus a visual overlay (box + confidence) for manual review; genuinely plate-less images went to a rejected folder. This recovered a meaningful chunk of data that a single-model pipeline would have thrown away.

### Stage 5 — Merge & split
Merged sources into a single dataset — **images symlinked** (avoids duplicating hundreds of thousands of files on disk), **labels copied** (small, and each split needs its own mutable copy).

Split 75/10/15 train/val/test with a **seeded shuffle**.

> **The shuffle is a real bug I caught — tell this story.** The merged dataset was in *merge order*: all of source A, then all of source B, then C. Splitting on that raw order would have put entire sources exclusively into train or test — so the test set would measure "can the model generalise to a camera it has never seen" rather than "how good is the model," and results would be wildly misleading. A seeded shuffle before splitting fixes it, and the fixed seed keeps it reproducible.

### Engineering practices worth mentioning
- **Checkpointing/resumability everywhere.** An accidental Ctrl-C at 75% of a multi-hour job taught me this the hard way. Every long-running stage now writes an append-only checkpoint of completed items and skips them on restart.
- **Thread-safe logging** — every decision (kept, rejected, reason) written to CSV under a lock.
- **Idempotency** — the plate-labelling script strips old class-4 lines before writing new ones, so re-running never double-adds boxes.
- **Failures are not checkpointed** — so a re-run automatically retries only what failed.
- **`screen` sessions** for long jobs so they survive SSH disconnects.

---

## 1.4 The models

### Model 1 — the 4-class vehicle detector
- YOLOv5-nano, 320×320, batch 256, 100 epochs, trained from COCO-pretrained `yolov5n.pt`.
- Measured **96.1% pure classification accuracy** on a held-out night dataset.

### Model 2 — the final 5-class model (current)
Fine-tuned from `plate_model_320.pt` — an existing **plate-only detector (nc=1)** — onto the full 5-class merged dataset at 640×640.

**Results at epoch 327/500:**

| Metric | Value |
|--------|-------|
| Precision | 98.08% |
| Recall | 97.27% |
| mAP@0.5 | 98.99% |
| mAP@0.5:0.95 | 93.78% |

---

## 1.5 The decisions you must be able to defend

These are the questions a sharp interviewer will ask. Have answers ready.

### "Why fine-tune instead of training from scratch?"
Transfer learning reuses learned low- and mid-level features — edges, textures, shapes — which are largely task-agnostic. Training from random init throws that away and needs far more data and epochs to reach the same point. With a nano model (low capacity, slower to converge cold) and a large but noisy dataset, the warm start was clearly the better bet.

### "How does YOLOv5 handle going from 1 class to 5?"
The detection head's output channels are a function of `nc` — specifically `(nc + 5) × num_anchors` per scale, where the 5 is `x, y, w, h, objectness`. So when the class count changes, those final conv layers no longer match in shape.

YOLOv5's `train.py` handles this with `intersect_dicts()`: it loads the checkpoint's state dict, keeps only tensors whose **shapes match** the new model, and loads with `strict=False`. Mismatched head layers are silently dropped and randomly reinitialised; the backbone and neck transfer intact.

**In my run the log said `Transferred 343/349 items`** — 343 tensors carried over, 6 were the reinitialised head. Quote that number; it proves you actually read your logs.

### "Why 640×640 when the previous model was 320?"
Number plates are **small objects** — often a tiny fraction of total image area. At 320×320 a plate can shrink to a handful of pixels after downsampling, below what the network can reliably detect. Doubling input resolution roughly quadruples the pixel area available for small objects. The cost is ~4× the compute and memory per image, which is why I halved the batch size.

**Follow-up they might ask — "what else helps small objects?"** Higher resolution, using the higher-resolution feature maps (P2 head), tiling large images, mosaic augmentation, anchor recomputation, and loss weighting.

### "Why lower the learning rate to 0.001?"
Default `lr0` for training from scratch is 0.01. When fine-tuning, a high LR in the first epochs will destroy the pretrained features before they can be usefully adapted — you effectively get scratch training with extra steps. Dropping 10× lets the network adjust gently. This is the single most important hyperparameter change when fine-tuning.

### "Why `iou_t: 0.15` instead of 0.20?"
`iou_t` is the IoU threshold for deciding which anchors count as **positive matches** during training. Small objects — like plates — struggle to exceed a strict threshold against any anchor, so they generate too few positive training signals. Loosening it gives small-object classes more supervision per epoch.

### "Do you need to hand-tune the loss gains for the new class count and resolution?"
No — YOLOv5 auto-scales them:
```python
hyp["box"] *= 3 / nl                        # scale to number of detection layers
hyp["cls"] *= nc / 80 * 3 / nl              # scale to class count
hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl   # scale to image area
```
Knowing this off the top of your head is impressive.

### "Why start from the plate model and not your own vehicle model?"
**Be honest — this is a great answer because it shows judgement under uncertainty.**

Say: *"Analytically the vehicle model was the safer base — it was validated at 96% accuracy, and its 4 classes were unchanged, so only one new class needed learning. Starting from the plate-only model meant all four vehicle classes trained essentially cold. I went ahead with the plate base as a deliberate experiment, and it worked — the merged dataset had enough labelled vehicle examples that cold-starting those classes wasn't a problem. The result was 93.8% mAP@0.5:0.95. But if data had been scarcer, the vehicle base would have been the right call."*

That answer shows you can reason about a trade-off, make a call, and evaluate it honestly afterwards.

### "How do you know the model is actually good?"
Two separate things, and **conflating them is a common junior mistake**:

- **mAP** measures detection quality — box localisation *and* classification together.
- **Classification accuracy** measures only "given that there's a vehicle here, did you name it correctly?"

I built a custom evaluation script for the second: match each ground-truth box to its best-overlapping prediction at a **deliberately loose IoU of 0.3** (I don't care about box tightness for this metric — I only need to link GT to prediction), then compare classes and build a confusion matrix. That gave 96.1% pure classification accuracy, separate from box quality.

I also built an error-analysis tool that scores each image by `false_negatives + false_positives + class_mismatches`, sorts by that score, and writes out the worst images with **green ground-truth boxes and red prediction boxes** overlaid — so I could actually *look* at the failures instead of guessing.

> **This is your strongest interview material.** Most candidates report one mAP number. Very few build custom metrics because they identified that the standard metric didn't answer their actual question.

### "How do you know it isn't overfitting?"
Validation losses tracked *down* in step with training losses across all 327 epochs — never diverged. Classic overfitting shows train loss falling while val loss rises. Also used early stopping (`patience=50`) so the run halts if fitness stops improving.

### "Read me your training curve."
- Epochs 0–40: mAP@0.5:0.95 shot from 0.486 → 0.894. Fast early learning.
- Epochs 40–150: 0.894 → 0.930. Diminishing returns.
- Epochs 150–327: 0.930 → 0.938. mAP@0.5, precision, recall had all plateaued (~0.990 / 0.981 / 0.972); only the **strict** metric was still creeping up — meaning the model had found essentially all the objects and was now just tightening box localisation.

---

## 1.6 Weaknesses — name them before they do

Interviewers respect self-awareness far more than false confidence.

1. **Labels are model-generated, not human-verified.** Ground truth inherits the biases and errors of the labelling models. Some portion of "errors" in evaluation may actually be label noise. I mitigated by sampling and visually reviewing overlays, but a human-annotated gold-standard test set would be the proper fix.

2. **Validation labels come from the same weak-supervision source as training labels.** So the model can be "correct" by reproducing a systematic labelling error. This means my metrics are probably slightly optimistic. The fix is a small hand-labelled holdout.

3. **Orientation was assigned per-batch, not predicted per-image.** Works when a camera only ever sees one side; breaks on mixed batches — which is exactly the problem I hit with the third dataset and haven't fully solved yet.

4. **The mixed day/night batch is unresolved.** Neither the front-only nor back-only labelling script applies, and dark images make orientation genuinely ambiguous.

5. **No real-world latency benchmark yet.** I know the architecture is nano-sized (1.77M params, 4.2 GFLOPs) but haven't measured end-to-end throughput on production hardware.

6. **Class imbalance not explicitly handled** — `reg_plate` appears in nearly every image while vehicle classes are split four ways.

---

## 1.7 If asked "what would you do next?"

- Hand-label a 1–2k image gold-standard test set for honest metrics.
- Solve mixed-orientation labelling properly — likely a small dedicated orientation classifier trained on the confidently-labelled subset.
- Benchmark inference latency and try INT8 quantisation / TensorRT export for deployment.
- Per-class metric breakdown to see specifically where `reg_plate` recall sits versus the vehicle classes.
- Active learning: use model uncertainty to pick which images are worth human labelling.

---

# PART 2 — DEEP LEARNING FOR INTERVIEWS

## 2.1 Fundamentals

**Neural network** — layers of weighted sums followed by non-linear activations. Without non-linearity, stacking layers collapses to a single linear transform, so depth would buy nothing.

**Forward pass** — input flows through the layers to produce a prediction.

**Loss function** — a single number measuring how wrong the prediction is.

**Backpropagation** — the chain rule applied efficiently. Compute the gradient of the loss with respect to every weight by propagating derivatives backwards through the network.

**Gradient descent** — nudge each weight in the direction that reduces loss: `w ← w − lr × ∂L/∂w`.

**Epoch / batch / iteration** — an epoch is one full pass over the dataset; a batch is the group of samples processed before one weight update; an iteration is one such update.

### Activation functions
| Function | Notes |
|----------|-------|
| **ReLU** `max(0,x)` | Default. Cheap, avoids vanishing gradients. Can "die" (stuck at 0). |
| **Leaky ReLU** | Small negative slope fixes dying ReLU. |
| **Sigmoid** | Squashes to (0,1). Used for binary/multi-label output. Saturates → vanishing gradients. |
| **Softmax** | Turns logits into a probability distribution summing to 1. Multi-class output. |
| **SiLU/Swish** | `x·sigmoid(x)`. What YOLOv5 actually uses. Smooth, usually slightly better than ReLU. |

### Loss functions
- **Cross-entropy** — classification. Heavily penalises confident wrong answers.
- **BCE (binary cross-entropy)** — multi-label, and what YOLO uses for objectness and class.
- **MSE / L1** — regression.
- **IoU-family (GIoU, DIoU, CIoU)** — bounding-box regression. Better than raw L1 on coordinates because it optimises the actual overlap metric you care about.
- **Focal loss** — down-weights easy examples so the model focuses on hard ones. Designed for extreme class imbalance in dense detection.

---

## 2.2 CNNs

**Convolution** — slide a small learned kernel over the image; each output pixel is a weighted sum of a local neighbourhood.

**Why convolutions instead of fully-connected layers for images?**
1. **Parameter sharing** — the same kernel is reused across every position, so you learn "edge detector" once instead of separately for every pixel location.
2. **Locality** — nearby pixels are correlated; distant ones usually aren't.
3. **Translation equivariance** — a car shifted right produces the same features, shifted right.

A fully-connected layer on a 640×640×3 image would need ~1.2M weights *per neuron*. Infeasible.

**Key terms:** *kernel/filter* (learned weights), *stride* (step size — stride 2 halves resolution), *padding* (preserve spatial size), *receptive field* (region of input influencing one output — grows with depth), *pooling* (downsample; max-pooling keeps the strongest activation).

**Feature hierarchy** — early layers learn edges and colours; middle layers learn textures and parts; deep layers learn object-level concepts. **This is exactly why transfer learning works** — the early layers are useful for almost any vision task.

---

## 2.3 Object detection

**Classification** = what is in this image. **Detection** = what, *and where*, and *how many*.

### Two-stage vs one-stage
- **Two-stage (R-CNN, Fast/Faster R-CNN)** — first propose candidate regions, then classify each. More accurate historically, slower.
- **One-stage (YOLO, SSD, RetinaNet)** — predict boxes and classes directly from feature maps in one pass. Faster, and the gap in accuracy has largely closed.

**YOLO = You Only Look Once.** One forward pass produces all boxes.

### Anchor boxes
Pre-defined box shapes at each grid cell. Rather than predicting box coordinates from nothing, the network predicts *offsets* from these priors — an easier learning problem. YOLOv5 uses 3 anchors × 3 scales = 9. **Autoanchor** runs k-means on your dataset's box shapes at training start and recomputes the anchors to fit your data — important when your objects (like plates) have unusual aspect ratios.

### NMS (Non-Maximum Suppression)
Detectors produce many overlapping boxes for the same object. NMS keeps the highest-confidence box, deletes every box overlapping it above an IoU threshold, and repeats. Controlled by the `iou` parameter at inference.

### IoU (Intersection over Union)
`area of overlap / area of union`. 0 = no overlap, 1 = perfect. Used both for NMS and for deciding whether a prediction "counts" as correct.

### Detection metrics
- **Precision** = TP / (TP + FP) — of what I predicted, how much was right? *Low precision = too many false alarms.*
- **Recall** = TP / (TP + FN) — of what exists, how much did I find? *Low recall = missing things.*
- **AP (Average Precision)** — area under the precision-recall curve for one class.
- **mAP** — AP averaged over classes.
- **mAP@0.5** — a prediction counts as correct at IoU ≥ 0.5. Lenient.
- **mAP@0.5:0.95** — averaged over IoU thresholds 0.5, 0.55 … 0.95. Strict; rewards tight boxes. **This is the headline number.**

> **Guaranteed question: "precision vs recall trade-off?"** Lowering the confidence threshold catches more true objects (↑recall) but also more junk (↓precision). Which you favour depends on the application — a medical screening tool wants recall; a spam filter wants precision.

### YOLOv5 architecture (three parts)
- **Backbone** (CSPDarknet) — extracts features, progressively downsampling.
- **Neck** (PANet/FPN) — fuses features across scales so small objects (fine, high-res features) and large objects (coarse, semantic features) are both detectable.
- **Head** — outputs `(nc + 5) × anchors` channels per scale: box coords, objectness, class scores.

**Model sizes:** n (nano) < s < m < l < x. Nano is ~1.8M params — chosen here for edge deployment.

---

## 2.4 Training concepts

### Overfitting vs underfitting
- **Overfitting** — memorises training data, fails to generalise. *Signal: train loss ↓ while val loss ↑.* Fixes: more data, augmentation, dropout, weight decay, early stopping, smaller model.
- **Underfitting** — too simple to capture the pattern. Both losses stay high. Fixes: bigger model, train longer, higher LR, less regularisation.

### Regularisation
- **L2 / weight decay** — penalises large weights, encourages simpler functions.
- **Dropout** — randomly zeroes neurons during training; prevents co-adaptation.
- **Batch normalisation** — normalises layer inputs per batch. Stabilises and accelerates training, mild regularisation effect.
- **Early stopping** — halt when validation stops improving. (Your `--patience 50`.)
- **Data augmentation** — the most effective one in vision.

### Augmentation (the ones in your hyp file)
| Param | What it does |
|-------|-------------|
| `hsv_h/s/v` | Colour jitter — robustness to lighting/camera differences. Critical for day+night data. |
| `fliplr: 0.5` | Horizontal flip half the time. |
| `flipud: 0.0` | Vertical flip **off** — vehicles are never upside-down; it would teach nonsense. |
| `translate`, `scale` | Position and size jitter. |
| `mosaic: 1.0` | Stitches 4 images into one. Big win for small-object detection — creates scale variety and more objects per batch. |
| `mixup`, `copy_paste` | Off — more useful for extreme imbalance / segmentation. |

> **Good answer to "how did you pick augmentations?":** *"Based on what's physically plausible for the data. Horizontal flip is fine — a car facing left is a real thing. Vertical flip isn't, so I disabled it. Heavy HSV augmentation because the dataset spans day and night with different cameras."*

### Optimisers
- **SGD** — plain gradient descent. With **momentum** (0.937 in your config) it accumulates a velocity term, smoothing updates and helping escape shallow local minima.
- **Adam / AdamW** — adaptive per-parameter learning rates. Converges faster; sometimes generalises slightly worse than tuned SGD. AdamW fixes Adam's weight-decay handling.

### Learning rate schedule
- **Warmup** (`warmup_epochs: 3`) — start tiny and ramp up. Early gradients are large and noisy; a full-size LR immediately can destabilise training, especially with freshly-initialised layers.
- **Decay** — `lrf: 0.01` means the final LR is 1% of the initial. Large steps early to explore, small steps late to settle into a minimum.

### Transfer learning
Take a model trained on a large dataset, reuse its weights for a new task. Options:
1. **Feature extraction** — freeze the backbone, train only the head. Best when new data is small.
2. **Full fine-tuning** — train everything at a low LR. Best when you have plenty of new data. **This is what I did.**

**Why it works:** early-layer features (edges, textures) are nearly universal across vision tasks.

### Mixed precision (AMP)
Uses FP16 for most operations and FP32 where precision matters, with **loss scaling** to prevent small gradients underflowing to zero in FP16. Roughly 2× faster and halves memory on modern GPUs. Your log's `AMP: checks passed ✅` confirms it was active.

### Batch size effects
Larger batches → more stable gradients, better GPU utilisation, but more memory and sometimes slightly worse generalisation. When you raise input resolution, memory per image scales ~quadratically — which is exactly why 320→640 forced batch 256→128.

---

## 2.5 Rapid-fire Q&A

**Why normalise inputs?** Keeps features on comparable scales so gradients are well-conditioned; unnormalised inputs make optimisation slow and unstable.

**Vanishing gradients?** In deep networks, repeatedly multiplying small derivatives shrinks gradients toward zero, so early layers stop learning. Mitigations: ReLU, batch norm, residual/skip connections.

**Train / validation / test?** Train fits the weights. Validation tunes hyperparameters and triggers early stopping. Test is touched **once**, at the end, for an unbiased estimate. If you tune against test, you've leaked and your number is optimistic.

**Data leakage?** Information from validation/test influencing training. Your near-duplicate problem is a perfect concrete example — the same photo in both train and test makes the model look better than it is. This is *why* deduplication came before splitting.

**Class imbalance fixes?** Weighted loss, oversampling the minority, undersampling the majority, focal loss, or targeted augmentation.

**Precision-recall vs ROC?** PR curves are more informative under heavy class imbalance, because ROC's false-positive rate is diluted by a huge negative class.

**Epoch vs iteration?** Epoch = full dataset pass. Iteration = one weight update = one batch.

**What's a confusion matrix?** Rows = true class, columns = predicted class. Diagonal = correct. Off-diagonal cells show *which* classes get mistaken for which — far more actionable than a single accuracy number.

---

# PART 3 — SKILLS TO LEARN

## 3.1 Solidify what you've already touched
You've used these — now understand them properly.

- **PyTorch fundamentals.** Write a training loop from scratch: dataset, dataloader, model, loss, optimiser, backward, step. You've been using YOLOv5's loop; build your own once and everything demystifies.
- **NumPy / OpenCV.** Array indexing, broadcasting, image ops, colour spaces.
- **Reading YOLOv5's source.** You've already read `train.py` — keep going into `utils/loss.py` and `utils/dataloaders.py`.
- **Git.** Non-negotiable for any engineering role.
- **Linux + bash.** You're already living in `screen`, `ssh`, `nvidia-smi`, pipes. Keep building.

## 3.2 Next tier (highest return on effort)

1. **Build one model end-to-end in raw PyTorch** — even a small CNN on CIFAR-10. This is the single best thing you can do. It converts "I ran a training script" into "I understand training."
2. **Deployment: ONNX → TensorRT, quantisation.** Where CV projects actually deliver value. Export your model, benchmark FP32 vs FP16 vs INT8 latency. Enormous interview differentiator.
3. **A proper experiment tracker** — Weights & Biases or MLflow. You've used TensorBoard; W&B is the industry norm.
4. **Docker.** Containerise your inference service. Every production ML job asks about this.
5. **FastAPI.** Wrap the model in a REST endpoint — you've been *consuming* an ANPR HTTP service; build one yourself.

## 3.3 Broader ML knowledge

- **Classical ML** — logistic regression, decision trees, random forests, gradient boosting (XGBoost). Interviewers ask about these even for DL roles.
- **Evaluation & statistics** — bias-variance, cross-validation, significance.
- **Transformers & attention** — ViT, DETR. The field's centre of gravity; you should at least be able to explain self-attention.
- **Segmentation** — semantic vs instance, U-Net, Mask R-CNN.
- **Self-supervised / contrastive learning** — how modern models learn from unlabelled data. Directly relevant to your weak-supervision problem.

## 3.4 Recommended order (next ~6 months)

| Phase | Focus |
|-------|-------|
| **1** | Raw-PyTorch CNN on CIFAR-10. Write every line yourself. |
| **2** | ONNX export + TensorRT + latency benchmarking on *your* model. |
| **3** | FastAPI + Docker: package your 5-class model as a service. |
| **4** | W&B for a proper hyperparameter sweep. |
| **5** | Read the YOLO papers (v1 → v3, then YOLOv5's design notes). |
| **6** | Attention & transformers: "Attention Is All You Need", then ViT and DETR. |

---

# PART 4 — INTERVIEW STRATEGY

## The STAR structure for your project story
- **Situation** — parking system, hundreds of thousands of unlabelled images, need vehicle type + orientation + plate location.
- **Task** — build the whole pipeline solo: data → labels → model → evaluation.
- **Action** — dedup at scale (BK-tree), weak-supervision labelling, multi-stage quality filtering, transfer learning from a plate-only model to 5 classes at higher resolution, custom evaluation tooling.
- **Result** — 98% precision, 97% recall, 93.8% mAP@0.5:0.95, single model replacing what would have been several.

## Numbers to have memorised
- ~460,000 images across 3 batches
- 5 classes
- Precision 98.08% · Recall 97.27% · mAP@0.5 98.99% · mAP@0.5:0.95 93.78%
- `Transferred 343/349 items` on the class-change fine-tune
- 96.1% pure classification accuracy on the earlier 4-class model
- YOLOv5-nano: 1.77M parameters, 4.2 GFLOPs, 214 layers
- 320×320 → 640×640; batch 256 → 128; `lr0` 0.01 → 0.001

## Three things that make you stand out
1. **You built custom evaluation tooling** because you identified that mAP didn't answer your actual question. Very few juniors do this.
2. **You caught the merge-order splitting bug.** It's subtle, it would have silently produced misleading metrics, and catching it shows real understanding of what a validation set is *for*.
3. **You thought about scale.** BK-tree instead of O(n²), symlinks instead of copies, checkpointing, batched GPU inference, reduced-resolution reads for brightness scoring.

## Rules
- **Never bluff.** "I haven't worked with that, but here's how I'd approach it" beats a wrong confident answer every single time.
- **Lead with the trade-off.** "I chose X over Y because…" is the sentence pattern of an engineer rather than a script-runner.
- **Bring the failures.** The Ctrl-C that cost 75% of a job (→ built checkpointing), the class-name bug, the merge-order bug. Interviewers trust people who debug honestly.
- **Ask them questions.** What does their data pipeline look like? How do they handle labelling? Do they have a human-verified test set?

---

*Good luck. You have a genuinely strong project — the depth of the data engineering is unusual for an intern, and the custom evaluation work is the kind of thing that makes an interviewer sit up.*
