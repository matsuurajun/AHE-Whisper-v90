# AHE-Whisper — Changelog


## [v90.97]— 2025-11-12
 * Added Smooth Aligner (α=0.40, γ=1.4) — fully stable across 3 runs (Jaccard=0.99)

## [v90.96] — 2025-11-10

**Experimental Branch — `feature/v90.96_local-tau-smooth-align`**

🧩 *Core*
* Introduced **local τ (temperature) scheduler** for dynamic softmax scaling based on similarity variance.
* Added **post-sharping (γ=1.3)** and entropy monitoring for stable speaker probability contrast.
* Integrated **probability smoothing (EMA / local window)** to reduce speaker-flutter and over-segmentation.
* Implemented **temporary storage & diagnostic release** (`last_probs`) for memory-efficient debugging.

⚙️ *Perf*
* Improved diarization precision while maintaining RTF ≈ 0.19.
* Aligner now exhibits natural speaker transitions comparable to v34.02, with enhanced stability.

📦 *Infra*
* Added early-memory release hook (`del self.last_probs`) to limit peak memory footprint.
* Branch: `feature/v90.96_local-tau-smooth-align`


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
