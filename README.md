# AHE-Whisper

**Adaptive Hybrid Engine for Whisper — v90.90 (Definitive Release)**
Author: Matsuura Jun
Date: November 2025

---

## 🧭 Overview

**AHE-Whisper** is an Apple-Silicon-optimized, end-to-end transcription and speaker-diarization engine.
It integrates multiple ASR, VAD, and embedding pipelines — each modular and swappable — to achieve **maximum speed, accuracy, and reproducibility** for TV production workflows such as *NHK World’s BIZ STREAM*.

---

## ⚙️ Core Components

| Module        | Role                            | Model                                  |
| ------------- | ------------------------------- | -------------------------------------- |
| **ASR**       | Speech recognition              | Whisper Large-v3-Turbo (MLX-optimized) |
| **VAD**       | Voice activity detection        | Silero-VAD (ONNX)                      |
| **Embedding** | Speaker representation          | WeSpeaker ECAPA-TDNN-512               |
| **Aligner**   | OverlapDPAligner (adaptive)     | Custom                                 |
| **Clusterer** | Attractor-based deep clustering | Experimental                           |
| **Frontend**  | niceGUI-based web UI (v9.x)     | Python 3.12 / MLX stack                |

---

## 🧩 Directory Structure

```
AHE-Whisper/
├── ahe_whisper/               # Core engine modules
├── tools/                     # Benchmarking, diagnostics
├── models/                    # (ignored) local model cache
├── setup_ahe_whisper.py       # Project generator & environment setup
├── requirements.txt           # Dependencies
├── run_offline.command        # macOS launch script
├── README.md                  # ← You are here
└── CHANGELOG.md               # Version history
```

> ⚠️  Note: The `models/` directory is excluded via `.gitignore`
> because of file size and licensing. Use `prefetch_models.py` or
> `setup_ahe_whisper.py` to auto-download required models.

---

## 🚀 Setup (Apple Silicon)

```bash
cd AHE-Whisper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_ahe_whisper.py --init
```

or (if using **uv**):

```bash
uv venv
uv pip install -r requirements.txt
uv run setup_ahe_whisper.py --init
```

---

## 🧠 Key Features

* 🔹 MLX-accelerated Whisper inference (M-series optimized)
* 🔹 Adaptive Hybrid Engine: automatic VAD × ASR integration
* 🔹 OverlapDPAligner + Attractor clustering for overlapping speech
* 🔹 Speaker-aware transcription export (`.srt`, `.vtt`, `.json`)
* 🔹 Deterministic reproducibility via `setup_ahe_whisper.py`
* 🔹 Offline execution (no external API dependency)

---

## 🧪 Typical Usage

```bash
python setup_ahe_whisper.py --transcribe input_audio.mp3
```

Outputs:

```
AHE-Whisper-output/
 ├── transcript.json
 ├── transcript.vtt
 └── speaker_timeline.csv
```

---

## 📄 License

All model files follow their respective upstream licenses (OpenAI, ONNX-Community, WeSpeaker).
Custom code © 2025 Matsuura Jun.
This repository is intended for internal R&D use and not for model redistribution.

---

## 🏷️ Version

Current Release: **v90.90 – Definitive**
Tag: `v90.90`
Date: 2025-11-08
Next planned milestone: **v91.00 (Fluid Inference Beta)**

