# -*- coding: utf-8 -*-
import time
import os
import logging
from pathlib import Path
import numpy as np
import librosa
from typing import Dict, Any

import mlx_whisper

from ahe_whisper.config import AppConfig
from ahe_whisper.embedding import build_er2v2_session, warmup_er2v2, er2v2_embed_batched
from ahe_whisper.vad import VAD
from ahe_whisper.diarizer import Diarizer
from ahe_whisper.aligner import OverlapDPAligner
from ahe_whisper.utils import get_metrics, add_metric, calculate_coverage_metrics
from ahe_whisper.word_grouper import group_words_sudachi
from ahe_whisper.model_manager import ensure_model_available

LOGGER = logging.getLogger("ahe_whisper_worker")

def run(
    audio_path: str,
    config: AppConfig,
    project_root: Path,
    stage: str = None
) -> Dict[str, Any]:
    
    t0 = time.perf_counter()
    waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_sec = len(waveform) / sr
    LOGGER.info(f"Audio loaded: duration={duration_sec:.2f}s")
    
    # --- Stage-selective partial run (for performance profiling) ---
    if stage:
        LOGGER.info(f"[PIPELINE] Running stage='{stage}' only for timing test")

        if stage == "asr":
            asr_model_path = ensure_model_available('asr', project_root)
            LOGGER.info("[STAGE-ASR] Loading Whisper model...")
            _ = mlx_whisper.transcribe(
                audio=waveform,
                path_or_hf_repo=str(asr_model_path),
                language=config.transcription.language,
                word_timestamps=True,
                no_speech_threshold=getattr(config.transcription, "no_speech_threshold", 0.65),
                condition_on_previous_text=False,
            )
            return {"stage": "asr", "ok": True}

        elif stage == "vad":
            vad_model_path = ensure_model_available('vad', project_root)
            LOGGER.info("[STAGE-VAD] Running Silero-VAD detection...")
            vad = VAD(vad_model_path, config.vad)
            _ = vad.get_speech_probabilities(waveform, sr, config.aligner.grid_hz)
            return {"stage": "vad", "ok": True}

        elif stage == "diar":
            ecapa_model_path = ensure_model_available('embedding', project_root)
            LOGGER.info("[STAGE-DIAR] Running ERes2NetV2 embedding extraction...")
            
            # --- ER2V2 セッション作成 ---
            er2_sess = build_er2v2_session(ecapa_model_path, config.embedding)
            warmup_er2v2(er2_sess)
            
            win_len = int(config.embedding.embedding_win_sec * sr)
            hop_len = int(config.embedding.embedding_hop_sec * sr)
            audio_chunks = [waveform[i:i+win_len] for i in range(0, len(waveform), hop_len)]
            
            # --- ER2V2 embedding 抽出 ---
            _ = er2v2_embed_batched(er2_sess, ecapa_model_path, audio_chunks, sr, config.embedding)
            
            return {"stage": "diar", "ok": True}

        else:
            LOGGER.warning(f"[PIPELINE] Unknown stage='{stage}', skipping execution.")
            return {"stage": stage, "ok": False}
    
    asr_model_path = ensure_model_available('asr', project_root)
    vad_model_path = ensure_model_available('vad', project_root)
    ecapa_model_path = ensure_model_available('embedding', project_root)
    
    er2_sess = build_er2v2_session(ecapa_model_path, config.embedding)
    warmup_er2v2(er2_sess)
    
    LOGGER.info(f"[DEBUG] calling mlx_whisper.transcribe: "
            f"no_speech_threshold={config.transcription.no_speech_threshold}, "
            f"vad_filter={config.transcription.vad_filter if hasattr(config.transcription, 'vad_filter') else 'N/A'}")

    asr_result = mlx_whisper.transcribe(
        audio=waveform,
        path_or_hf_repo=str(asr_model_path),
        language=config.transcription.language,
        word_timestamps=True,
        no_speech_threshold=getattr(config.transcription, "no_speech_threshold", 0.65),
        condition_on_previous_text=False,
)
    LOGGER.info(f"[DEBUG] asr_result keys: {list(asr_result.keys()) if asr_result else 'EMPTY'}")
    
    if asr_result and "segments" in asr_result:
        total_words = sum(len(seg.get("words", [])) for seg in asr_result["segments"])
        LOGGER.info(f"[TRACE-ASR-RAW] segments={len(asr_result['segments'])}, total_words={total_words}")
        
    segments = asr_result.get("segments", []) or []
    words = []

    # --- ASR 結果の正規化（segments→words） ---
    for seg in segments:
        seg_start = float(seg.get("start", 0.0) or 0.0)
        seg_end   = float(seg.get("end",   seg_start) or seg_start)
        
        if seg_end <= seg_start:
            continue
        
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                w_start = float(w.get("start", seg_start) or seg_start)
                w_end   = float(w.get("end",   seg_end)   or seg_end)
                if w_end <= w_start:
                    continue
                words.append({
                    "word":  (w.get("word") or w.get("text") or "").strip(),
                    "start": w_start,
                    "end":   w_end,
                    "confidence": w.get("confidence"),
                    "avg_logprob": w.get("avg_logprob"),
                })
        else:
            # word_timestamps が無い場合はセグメント単位で補完
            text = (seg.get("text") or "").strip()
            words.append({
                "word":  text,
                "start": seg_start,
                "end":   seg_end,
                "confidence": seg.get("confidence"),
                "avg_logprob": seg.get("avg_logprob"),
            })

    # --- Debug 出力（Aligner に渡す直前ログ） ---
    if words:
        first_w = words[0]; last_w = words[-1]
        LOGGER.info("[DEBUG-ASR→ALIGN] words_len=%d, first=(%.2f,'%.20s'), last=(%.2f,'%.20s')",
                    len(words),
                    float(first_w.get("start", 0.0) or 0.0), str(first_w.get("word",""))[:20],
                    float(last_w.get("end", 0.0) or 0.0),   str(last_w.get("word",""))[:20])
    else:
        LOGGER.warning("[DEBUG-ASR→ALIGN] words_len=0 (flatten failed)")

    LOGGER.info(f"[DEBUG-ASR] segments={len(segments)}, words={len(words)}, dur={asr_result.get('duration', 'N/A')}s")

    vad = VAD(vad_model_path, config.vad)
    vad_probs, grid_times = vad.get_speech_probabilities(waveform, sr, config.aligner.grid_hz)
    add_metric("vad.grid_size", len(grid_times))
    calculate_coverage_metrics(words, vad_probs, duration_sec, config)

    if not words:
        LOGGER.warning("Whisper detected no words. Aborting diarization.")
        return {"words": [], "speaker_segments": [], "duration_sec": duration_sec, "metrics": get_metrics(), "is_fallback": True}
    
    asr_words = words

    win_len = int(config.embedding.embedding_win_sec * sr)
    hop_len = int(config.embedding.embedding_hop_sec * sr)
    audio_chunks = [waveform[i:i+win_len] for i in range(0, len(waveform), hop_len)]
    
    if audio_chunks:
        embeddings, valid_embeddings_mask = er2v2_embed_batched(
            er2_sess,
            ecapa_model_path,
            audio_chunks,
            sr,
            config.embedding
        )
    else:
        embeddings = np.zeros((0, config.embedding.embedding_dim), dtype=np.float32)
        valid_embeddings_mask = np.zeros(0, dtype=bool)
    
    is_fallback = False
    if not np.any(valid_embeddings_mask):
        LOGGER.warning("No valid embeddings extracted. Falling back to single speaker.")
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        speaker_segments = [(valid_words[0]['start'], valid_words[-1]['end'], 0)] if valid_words else []
        is_fallback = True
    else:
        diarizer = Diarizer(config.diarization)
        speaker_centroids, labels = diarizer.cluster(embeddings[valid_embeddings_mask])
        add_metric("diarizer.num_speakers_found", len(speaker_centroids))
        
        spk_probs = diarizer.get_speaker_probabilities(embeddings, valid_embeddings_mask, speaker_centroids, grid_times, hop_len, sr)
        
        # === [PATCH v90.95] 一時保存内容のログ確認と解放 ===
        try:
            if hasattr(diarizer, "last_probs"):
                LOGGER.info("[DEBUG-DIAR] last_probs available: shape=%s, mean_max=%.3f, entropy=%.3f",
                            str(diarizer.last_probs.shape),
                            float(np.mean(np.max(diarizer.last_probs, axis=1))),
                            float(-np.mean(np.sum(
                                diarizer.last_probs * np.log(np.clip(diarizer.last_probs, 1e-9, 1.0)), axis=1))))
                # --- release memory early (analysis aid) ---
                del diarizer.last_probs
                LOGGER.debug("[DEBUG-DIAR] last_probs deleted to reduce memory footprint")
        except Exception as e:
            LOGGER.warning(f"[DEBUG-DIAR] could not inspect/delete last_probs: {e}")
        
        # === Diagnostic and normalization enhancement ===
        LOGGER.info("[DEBUG-DIAR] valid_sims diagnostics before normalization")
        try:
            if hasattr(diarizer, "last_valid_sims"):
                sims = diarizer.last_valid_sims
                LOGGER.info("[DEBUG-DIAR] valid_sims stats: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                            float(np.min(sims)), float(np.max(sims)),
                            float(np.mean(sims)), float(np.std(sims)))
                if float(np.std(sims)) < 0.05:
                    LOGGER.warning("[DEBUG-DIAR] valid_sims has very low variance (%.4f). Applying contrast normalization.", float(np.std(sims)))
                    sims = (sims - np.mean(sims, axis=1, keepdims=True)) / (np.std(sims, axis=1, keepdims=True) + 1e-6)
                    sims = np.tanh(sims)
                    diarizer.last_valid_sims = sims  # overwrite for consistency
        except Exception as e:
            LOGGER.warning("[DEBUG-DIAR] could not inspect valid_sims: %s", str(e))

        # --- Improved normalization with contrast scaling + temperature ---
        tau = 0.4
        scale = 3.0  # <= 新規追加：分散を増幅
        
        # --- sanity log ---
        LOGGER.info("[SPK-PROBS-RAW] shape=%s, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                    str(spk_probs.shape), float(np.min(spk_probs)), float(np.max(spk_probs)), 
                    float(np.mean(spk_probs)), float(np.std(spk_probs)))
        
        spk_probs = (spk_probs + 1.0) / 2.0
        spk_probs = np.clip(spk_probs, 0.0, 1.0)
        
        # コントラスト強調
        #spk_probs = (spk_probs - 0.5) * scale + 0.5
        #spk_probs = np.clip(spk_probs, 0.0, 1.0)
        
        max_per_row = np.max(spk_probs, axis=1, keepdims=True)
        spk_probs = np.exp((spk_probs - max_per_row) / tau)
        spk_probs = spk_probs / np.sum(spk_probs, axis=1, keepdims=True)
        
        if np.any(~np.isfinite(spk_probs)):
            spk_probs = np.nan_to_num(spk_probs, nan=1.0 / spk_probs.shape[1])
        
        LOGGER.info("[SPK-PROBS] mean_max=%.3f, mean_entropy=%.3f (tau=%.2f, scale=%.1f)",
                    float(np.mean(np.max(spk_probs, axis=1))),
                    float(-np.mean(np.sum(spk_probs * np.log(np.clip(spk_probs, 1e-9, 1.0)), axis=1))),
                    tau, scale)
        
        # --- Mini Enhancement: Stabilize speaker transition detection ---
        # 強制的に話者確率分布にスパース性を導入
        entropy = -np.sum(spk_probs * np.log(spk_probs + 1e-8), axis=1)
        low_entropy_mask = entropy < 1.5  # 信頼度高い領域
        spk_probs[low_entropy_mask] *= 1.1
        spk_probs = np.clip(spk_probs, 0.0, 1.0)
        
        # 相対差が小さいフレームを減衰
        margin = np.max(spk_probs, axis=1) - np.partition(spk_probs, -2, axis=1)[:, -2]
        low_margin_mask = margin < 0.1
        spk_probs[low_margin_mask] *= 0.8

        # Aligner tuning for better switching
        config.aligner.delta_switch = 0.1
        config.aligner.beta = 0.3
        config.aligner.gamma = 0.5
        
        config.aligner.non_speech_th = 0.02
        aligner = OverlapDPAligner(config.aligner)
        
        # === DEBUG (AHE): pre-align check ===
        try:
            LOGGER.info(f"[TRACE-ALIGNER-PRECHECK] type(words)={type(words)}, "
                        f"len(words)={len(words) if hasattr(words,'__len__') else 'N/A'}")
            if isinstance(words, list):
                LOGGER.info(f"[TRACE-ALIGNER-PRECHECK] sample(0:3)={words[:3]}")
        except Exception as e:
            LOGGER.error(f"[TRACE-ALIGNER-PRECHECK] inspection failed: {e}")
        
        # === EXISTING LOG ===
        LOGGER.info(f"[TRACE-ALIGNER-IN] words={len(words)}, vad_probs={len(vad_probs)}, "
                    f"spk_probs={spk_probs.shape if hasattr(spk_probs, 'shape') else 'N/A'}, "
                    f"grid_times={len(grid_times)}")
        
        speaker_segments = aligner.align(words, vad_probs, spk_probs, grid_times)
        
        # --- normalize speaker_segments (tuple → dict) (★これを1回だけ) ---
        if speaker_segments and isinstance(speaker_segments[0], (list, tuple)):
            speaker_segments = [
                {"start": float(s), "end": float(e), "speaker": f"SPEAKER_{int(spk):02d}"}
                for s, e, spk in speaker_segments
            ]
            LOGGER.info(f"[PIPELINE-FIX] Normalized speaker_segments to dict list: {len(speaker_segments)} items")
        elif not speaker_segments:
            LOGGER.warning("[PIPELINE-FIX] speaker_segments is empty before merge")

        # --- NEW: post-merge for short segments (v34-like behavior) ---
        def merge_short_segments(segments, min_len=2.0):
            if not segments or not isinstance(segments[0], dict):
                return segments
            
            merged = []
            prev = segments[0]
            for cur in segments[1:]:
                if (prev["speaker"] == cur["speaker"]) and ((cur["end"] - prev["end"]) < 0.5):
                    # 同一スピーカーで 0.5秒未満の隙間 → マージ
                    prev["end"] = cur["end"]
                    continue
                
                # 短すぎるセグメントの場合 → 次の話者とマージ
                if (prev["end"] - prev["start"]) < min_len:
                    cur["start"] = min(prev["start"], cur["start"])
                else:
                    merged.append(prev)
                
                prev = cur
            
            # 最終セグメントチェック
            if (prev["end"] - prev["start"]) < min_len and merged:
                merged[-1]["end"] = prev["end"]
            else:
                merged.append(prev)
            
            return merged
        
        speaker_segments = merge_short_segments(speaker_segments, min_len=2.0)
        LOGGER.info(f"[POST-MERGE] segments after merge_short={len(speaker_segments)}")
        
        # --- ALIGNMENT SANITY CHECK: ended too early? ---
        if speaker_segments and speaker_segments[-1]["end"] < duration_sec * 0.9:
            LOGGER.warning(
                f"[ALIGNER-FIX] alignment ended early at {speaker_segments[-1]['end']:.1f}s (<90% of audio). Expanding fallback."
            )
            speaker_segments = [{"start": 0.0, "end": duration_sec, "speaker": "SPEAKER_00"}]

    words = group_words_sudachi(words)
    add_metric("asr.word_count", len(words))

    if not speaker_segments and words:
        LOGGER.warning("DP alignment yielded no segments. Falling back to a single speaker segment.")
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        speaker_segments = [(valid_words[0]['start'], valid_words[-1]['end'], 0)] if valid_words else []
        is_fallback = True

    add_metric("pipeline.total_time_sec", time.perf_counter() - t0)
    
    # --- reconstruct text for exporter if missing ---
    if "text" not in locals() or not isinstance(asr_result.get("text"), str) or not asr_result["text"].strip():
        reconstructed_text = " ".join([w["word"] for w in words if "word" in w])
        asr_result["text"] = reconstructed_text
        LOGGER.info(f"[PIPELINE-FIX] Reconstructed text length: {len(reconstructed_text)}")

    return {
        "words": words,
        "speaker_segments": speaker_segments,
        "duration_sec": duration_sec,
        "metrics": get_metrics(),
        "is_fallback": is_fallback
    }
