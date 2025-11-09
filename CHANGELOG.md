# AHE-Whisper — Changelog

## [v90.90] — 2025-11-08

**Definitive Release**

* Consolidated all modules into unified `setup_ahe_whisper.py`
* Introduced hybrid VAD-ASR inference with OverlapDPAligner
* Integrated WeSpeaker ECAPA-TDNN512 (ONNX) for embeddings
* Optimized MLX Whisper inference (RTF ≈ 0.27 on M4 24 GB)
* Improved deterministic session cache handling
* Finalized GUI layer (niceGUI v9.5)
* Added `.gitignore` for model and output exclusion
* Verified reproducibility across M-series Macs

---

## [v83.12] — 2025-09-14

* Added adaptive energy normalization for Silero-VAD
* Enhanced diarization stability on multi-speaker audio
* Introduced internal logging pipeline (`TRACE-ALIGNER` mode)

---

## [v75.02] — 2025-07-03

* Transitioned to ONNXRuntime for embedding inference
* Added batch segment alignment with DP post-processing

---

## [v71.00] — 2025-05-11

* Implemented MLX-based WhisperKit backend
* Introduced `prefetch_models.py` for offline model setup

---

## [v34.00] — 2024-11-20

* Initial integration of VAD × ASR hybrid pipeline
* Prototype GUI (v3.0) and config generator

---

### Legend

🧩 *Core* = major architecture or engine changes
⚙️ *Perf* = speed or memory optimization
🧠 *UX/UI* = interface or usability update
📦 *Infra* = environment or build-related change
