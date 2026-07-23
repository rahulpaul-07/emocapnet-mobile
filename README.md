# EmoCapNet-Mobile

**Emotion-controllable image captioning, compressed for on-device deployment via knowledge distillation + INT8 quantization.**

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-261230)](pyproject.toml)

Given an image plus a requested emotion (one of 7 classes: factual + Ekman's six) and a
valence/arousal/dominance (VAD) vector, the model generates a caption *in that emotional
register* — e.g. the same photo captioned factually, joyfully, or fearfully. This repo takes a
trained ~260M-parameter teacher (ViT-base + GPT-2 with cross-attention) and distills it into a
~90M student (TinyCLIP/MobileViT + DistilGPT-2) that quantizes to INT8 for CPU/mobile inference —
an end-to-end **~10x on-disk compression** with a measured, not assumed, accuracy delta.

```
                         ┌──────────────────────────────────────────────┐
                         │  TEACHER — EmoCapNet v3 (frozen, ~260M)      │
                         │  ViT-base ─► GPT-2 + cross-attention         │
                         └───────┬──────────────────────────────────────┘
                                 │ soft logits · VAD/emotion heads · image features
              KD (KL, T=2) ──────┼───────── feature alignment (cosine)
                                 ▼
 image ──► TinyCLIP-ViT-8M ─► FiLM(VAD) ─► visual prefix + <emo=…> seed
                                 │
 VAD ────► cond MLP ─────────────┤            STUDENT (~90M FP32 → ~25% INT8)
                                 ▼
           DistilGPT-2 + cross-attention ─► caption  ┐
                                 │                   ├─ LM / KD loss
                                 ├─► VAD head        ├─ VAD loss
                                 ├─► emotion head    ├─ emotion loss
                                 └─► img/cap proj    └─ contrastive (InfoNCE)
```

## Why this design

Shrinking a network usually costs accuracy. The fix is **knowledge distillation**: the trained
teacher stays frozen and the student learns to imitate the teacher's temperature-softened
next-token distributions *and* the ground truth (`α·KD + (1−α)·LM`), plus the teacher's VAD and
emotion heads, plus a cosine feature-alignment term that transfers what the larger ViT "sees"
into the small encoder. With `distill_alpha=0` the objective reduces exactly to the original v3
multi-task loss — distillation is a strict superset, not a fork.

Compression is **INT8 dynamic quantization** on Linear + Embedding layers (HF's `Conv1D` is
first converted to a numerically identical `nn.Linear` — verified lossless in tests). FP16 is
kept as a safety net. The KV-cached greedy decoder is the same code path used for the CPU
latency benchmark, so the reported ms/caption is the honest mobile number.

## Quickstart

```bash
git clone <this-repo> && cd emocapnet-mobile
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or your CUDA wheel
pip install -e ".[dev]"

# Prove the whole pipeline works in ~2 minutes on a laptop CPU — no data needed:
emocapnet smoke --out runs/smoke
```

`smoke` generates a synthetic dataset, trains a tiny student for one epoch, evaluates
BLEU/METEOR/ROUGE-L, quantizes to INT8, and benchmarks CPU latency. If it prints `SMOKE OK`,
every stage of the system is wired correctly.

### Training on real data

Data expectations: a captions CSV with columns
`image_id, emotion, caption, valence, arousal, dominance` and one or more image directories.
The teacher checkpoint (`best.pt` from EmoCapNet v3 training) enables distillation; without it
the student trains on ground truth only (the code degrades gracefully and says so).

```bash
emocapnet train --config configs/default.yaml \
    -o captions_csv=/path/to/captions.csv \
    -o image_dirs=/path/to/images \
    -o teacher_ckpt=/path/to/best.pt

emocapnet evaluate --config configs/default.yaml --controllability --eval-teacher
emocapnet quantize --config configs/default.yaml
```

On Kaggle, use `configs/kaggle.yaml` (the original dataset paths are preserved there).

### Inference API

```bash
pip install -e ".[serve]"
emocapnet serve --config configs/default.yaml --ckpt runs/default/checkpoints/best.pt --int8
```

```bash
# one caption in a requested emotion (VAD optional; defaults to the emotion's prior)
curl -F "image=@photo.jpg" -F "emotion=happiness" localhost:8000/caption
# {"caption": "...", "emotion": "happiness", "vad": {...}, "latency_ms": 61.0}

# all seven emotional registers for one image
curl -F "image=@photo.jpg" localhost:8000/caption/emotions
```

`GET /health` and `GET /info` cover liveness and model metadata; interactive docs at `/docs`.
`--int8` serves the dynamically quantized build — same code path measured in the latency
benchmark. Input validation (unknown emotion, partial/out-of-range VAD, non-image upload)
returns proper 4xx errors, all covered by tests.

### ONNX export

```bash
pip install -e ".[onnx]"
emocapnet export-onnx --config configs/default.yaml --out onnx/
```

Exports two graphs — `encoder.onnx` (pixels + VAD → visual prefix + cross-attention memory,
runs once per image) and `decoder.onnx` (token ids + memory → next-token logits, runs per
token) — plus `onnx_metadata.json` with the decoding hyperparameters. The export then runs a
**parity check**: an onnxruntime greedy decoder (same repetition-penalty / no-repeat-ngram /
min-length logic) must produce token-for-token identical output to the torch decoder, or the
command fails. The decoder graph is cache-less for portability; adding KV-cache I/O bindings
is the documented next step for production ORT-mobile latency.

### Docker

```bash
docker build -t emocapnet-mobile .
docker run --rm emocapnet-mobile smoke --out /tmp/smoke
```

## Repository layout

```
src/emocapnet/
├── config.py            # typed dataclass config, YAML + key=value overrides
├── constants.py         # emotion labels, special tokens
├── tokenization.py      # shared teacher/student vocabulary (KD alignment invariant)
├── data/
│   ├── datamodule.py    # load/clean/split (image-level, leak-free, seed-fixed)
│   ├── dataset.py       # dual-resolution dataset (teacher + student pixels)
│   └── synthetic.py     # hermetic dataset generator for tests/CI
├── models/
│   ├── teacher.py       # EmoCapNet v3 skeleton + strict checkpoint loading
│   ├── student.py       # mobile captioner + KV-cached greedy decoding
│   └── encoders.py      # TinyCLIP / MobileViT / TinyConv behind one interface
├── losses.py            # KD + LM + VAD + emotion + contrastive + feature alignment
├── training/            # trainer (Stage-0 pretrain, staged unfreeze, AMP), checkpoints
├── evaluation/          # BLEU/METEOR/ROUGE-L/CIDEr-D, VAD controllability, CPU latency
├── compression/         # Conv1D→Linear, INT8 dynamic quant, FP16, size accounting
├── serving/             # FastAPI inference API (/caption, /caption/emotions, /info)
└── export/              # ONNX encoder/decoder export + onnxruntime parity check
```

## Evaluation methodology

Three separate questions, three separate instruments:

1. **Caption quality** — BLEU-1/4, METEOR, ROUGE-L, CIDEr-D on a held-out, image-level split
   identical to the teacher's, so the teacher-vs-student delta is apples-to-apples. A
   per-emotion BLEU-4 breakdown is reported because emotion control is the point of the model.
2. **Controllability** — does a requested VAD actually steer the text? Measured as Pearson r
   between requested VAD and VAD read back from generated captions using the *independent*
   NRC-VAD lexicon (never our own `vad_head`, which would be circular).
3. **Efficiency** — params, on-disk size (FP32/FP16/INT8), and CPU ms/caption with the
   KV-cached decoder. FP32-vs-INT8 metrics are compared on the same subset to quantify the
   quantization hit.

Fill this table from your training run (`emocapnet evaluate` + `emocapnet quantize`):

| Model | Params | Disk | BLEU-4 | METEOR | ROUGE-L | CIDEr-D | CPU ms/cap |
|---|---|---|---|---|---|---|---|
| Teacher (v3, FP32) | ~260M | ~1 GB | — | — | — | — | — |
| Student (FP32) | ~90M | ~350 MB | — | — | — | — | — |
| Student (INT8) | ~90M | ~90 MB | — | — | — | — | — |

> **Comparing to published work:** stylized/emotional captioning papers (SentiCap, MemCap,
> FlickrStyle10K systems) use different datasets and 2–4 style classes versus 7 here, so treat
> any side-by-side as context, not a leaderboard.

## Engineering notes

- **Reproducibility** — single seed controls python/numpy/torch; splits are deterministic and
  written to disk; the resolved config is saved next to every run.
- **Fail-fast distillation** — a teacher checkpoint that doesn't match the v3 skeleton raises
  immediately rather than silently distilling from partly-random weights.
- **Hermetic CI** — unit tests + a full end-to-end smoke pipeline run on every push with a tiny
  conv encoder and scratch decoder (no pretrained downloads beyond the GPT-2 tokenizer).
- **Graceful degradation** — no teacher → ground-truth training; no CLIP download → MobileViT;
  no NRC-VAD lexicon → VADER valence-only probe; no pycocoevalcap → local CIDEr-D.
- **Deployment path** — `emocapnet serve` (FastAPI, FP32 or INT8) for server inference;
  `emocapnet export-onnx` (with enforced torch↔ORT parity) as the on-ramp to ONNX Runtime
  Mobile; ExecuTorch / TFLite remain the documented next hop for a native phone build.

## Development

```bash
make install-dev   # editable install with dev tools
make test          # fast unit tests
make test-all      # everything incl. end-to-end, with coverage
make lint          # ruff check + format check
make smoke         # hermetic pipeline check
```

## License

MIT — see [LICENSE](LICENSE).
