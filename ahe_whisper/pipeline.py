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
from ahe_whisper.post_diar import sanitize_speaker_timeline
from ahe_whisper.model_manager import ensure_model_available

LOGGER = logging.getLogger("ahe_whisper_worker")

def _smooth_embeddings_over_time(embeddings: np.ndarray, kernel: int) -> np.ndarray:
    """
    時間方向の移動平均で埋め込みのブレを軽く平滑化する。
    kernel は奇数推奨（例: 3,5）。
    """
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or kernel <= 1:
        return embeddings
    k = int(kernel)
    if k < 2:
        return embeddings
    radius = k // 2
    # 端は edge パディングで延長
    padded = np.pad(embeddings, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(embeddings)
    for i in range(embeddings.shape[0]):
        smoothed[i] = padded[i : i + k].mean(axis=0)
    return smoothed

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

    # --- Diarization config の実行時確認ログ ---
    LOGGER.info(
        "[CONFIG-DIAR] min_speakers=%d, max_speakers=%d",
        config.diarization.min_speakers,
        config.diarization.max_speakers,
    )
    
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
        return {
            "words": [],
            "speaker_segments": [],
            "duration_sec": duration_sec,
            "metrics": get_metrics(),
            "is_fallback": True,
        }
    
    asr_words = words

    win_len = int(config.embedding.embedding_win_sec * sr)
    hop_len = int(config.embedding.embedding_hop_sec * sr)
    audio_chunks = [waveform[i:i+win_len] for i in range(0, len(waveform), hop_len)]
    
    if audio_chunks:
        # --- VAD をチャンク単位に要約（平均 speech prob） ---
        chunk_speech_scores = []
        for idx in range(len(audio_chunks)):
            chunk_start_t = (idx * hop_len) / sr
            chunk_end_t = min(chunk_start_t + win_len / sr, duration_sec)

            frame_indices = [
                j for j, t in enumerate(grid_times)
                if (t >= chunk_start_t) and (t < chunk_end_t)
            ]
            if frame_indices:
                score = float(np.mean(vad_probs[frame_indices]))
            else:
                score = 0.0
            chunk_speech_scores.append(score)

        # --- ER2V2 embedding 抽出 ---
        embeddings, valid_embeddings_mask = er2v2_embed_batched(
            er2_sess,
            ecapa_model_path,
            audio_chunks,
            sr,
            config.embedding,
        )

        # --- OPTIONAL: 時間方向スムージングで「同一話者内のブレ」を抑制 ---
        smooth_k = int(getattr(config.embedding, "smooth_embeddings_kernel", 0) or 0)
        if smooth_k >= 3:
            before_shape = embeddings.shape
            embeddings = _smooth_embeddings_over_time(embeddings, smooth_k)
            LOGGER.info(
                "[EMB-SMOOTH] applied kernel=%d on embeddings shape=%s",
                smooth_k,
                before_shape,
            )

        # --- VAD ベースで「発話を含むチャンク」に絞り込む ---
        chunk_speech_scores = np.asarray(chunk_speech_scores, dtype=np.float32)
        vad_th = getattr(config.embedding, "min_chunk_speech_prob", 0.30)
        vad_mask = chunk_speech_scores >= vad_th

        if len(vad_mask) != len(valid_embeddings_mask):
            LOGGER.warning(
                "[EMB-VAD] vad_mask len(%d) != valid_embeddings_mask len(%d). "
                "Skipping VAD refinement.",
                len(vad_mask),
                len(valid_embeddings_mask),
            )
        else:
            before = int(np.sum(valid_embeddings_mask))
            valid_embeddings_mask = np.logical_and(valid_embeddings_mask, vad_mask)
            after = int(np.sum(valid_embeddings_mask))
            LOGGER.info(
                "[EMB-VAD] chunks=%d, valid_before=%d, vad_speech>=%.2f -> valid_after=%d",
                len(audio_chunks),
                before,
                vad_th,
                after,
            )
    else:
        embeddings = np.zeros((0, config.embedding.embedding_dim), dtype=np.float32)
        valid_embeddings_mask = np.zeros(0, dtype=bool)
    
    is_fallback = False
    num_speakers_found = 0
    if not np.any(valid_embeddings_mask):
        LOGGER.warning("No valid embeddings extracted. Falling back to single speaker.")
        valid_words = [
            w for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
        if valid_words:
            speaker_segments = [
                {
                    "start": float(valid_words[0]["start"]),
                    "end":   float(valid_words[-1]["end"]),
                    "speaker": "SPEAKER_00",
                }
            ]
            is_fallback = True
        else:
            speaker_segments = []
            is_fallback = True

    else:
        diarizer = Diarizer(config.diarization)
        emb_valid = embeddings[valid_embeddings_mask]
        speaker_centroids, labels = diarizer.cluster(emb_valid)

        # === DIAR stats: frame-level cluster mass & found speakers ===
        if labels.size == 0:
            num_speakers_found = 0
            cluster_mass: list = []
        else:
            unique_labels, counts = np.unique(labels, return_counts=True)
            num_speakers_found = int(unique_labels.size)
            total_frames = int(counts.sum())
            cluster_mass = (
                counts.astype(np.float32) / float(max(1, total_frames))
            ).tolist()

        add_metric("diarizer.num_speakers_found", num_speakers_found)
        add_metric("diarizer.cluster_mass", cluster_mass)
        add_metric("diarizer.cluster_count", num_speakers_found)

        LOGGER.info(
            "[DIAR-STATS] frame_clusters=%d, mass=%s",
            num_speakers_found,
            ", ".join(f"{m:.3f}" for m in cluster_mass) if cluster_mass else "[]",
        )

        spk_probs = diarizer.get_speaker_probabilities(
            embeddings,
            valid_embeddings_mask,
            speaker_centroids,
            grid_times,
            hop_len,
            sr,
        )

        # === [DIAG v91] speaker-probability health check (normalizationは diarizer 側で完結) ===
        try:
            max_per_row = np.max(spk_probs, axis=1)
            mm = float(np.mean(max_per_row))
            ent = float(
                -np.mean(
                    np.sum(
                        spk_probs * np.log(np.clip(spk_probs, 1e-9, 1.0)),
                        axis=1,
                    )
                )
            )
            LOGGER.info(
                "[SPK-PROBS] from diarizer: mean_max=%.3f, mean_entropy=%.3f, shape=%s",
                mm,
                ent,
                str(spk_probs.shape),
            )
        except Exception as e:
            LOGGER.warning(f"[SPK-PROBS] diagnostic failed: {e}")
               
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
        
        # --- NEW: smooth speaker "islands" (A-B-A patterns) ---
        def smooth_speaker_islands(segments, max_island_sec=None):
            """
            A-B-A 型の短い「話者島」を前後の話者に吸収して丸める。
            その後、同一話者で隙間 < 0.5秒 のセグメントをマージする。
            """
            if not segments or len(segments) < 3 or not isinstance(segments[0], dict):
                return segments

            # しきい値のデフォルト：config.diarization.max_island_sec があれば使う
            if max_island_sec is None:
                max_island_sec = float(
                    getattr(config.diarization, "max_island_sec", 2.0)
                )

            # dict をコピーして安全側に
            segs = [dict(s) for s in segments]

            # 1) A-B-A パターンの B を A で塗りつぶす
            for i in range(1, len(segs) - 1):
                prev_seg = segs[i - 1]
                cur_seg = segs[i]
                next_seg = segs[i + 1]

                prev_spk = prev_seg.get("speaker")
                cur_spk = cur_seg.get("speaker")
                next_spk = next_seg.get("speaker")

                if not prev_spk or not next_spk:
                    continue
                # A-B-A のときだけ対象
                if prev_spk != next_spk or prev_spk == cur_spk:
                    continue

                dur = float(cur_seg["end"]) - float(cur_seg["start"])
                if dur <= max_island_sec:
                    # 真ん中の短い島を前後の話者で塗りつぶす
                    cur_seg["speaker"] = prev_spk

            # 2) 隣接する同一話者セグメントをマージ（隙間 < 0.5秒）
            merged = []
            for seg in segs:
                if not merged:
                    merged.append(seg)
                    continue

                last = merged[-1]
                if (
                    seg.get("speaker") == last.get("speaker")
                    and float(seg["start"]) - float(last["end"]) < 0.5
                ):
                    # 同一話者で 0.5秒未満の隙間 → 1つにまとめる
                    last["end"] = float(seg["end"])
                else:
                    merged.append(seg)

            return merged

        # diarization.min_speaker_duration_sec をしきい値として利用
        min_len = float(
            getattr(config.diarization, "min_speaker_duration_sec", 1.5)
        )
        speaker_segments = merge_short_segments(speaker_segments, min_len=min_len)
        LOGGER.info(
            "[POST-MERGE] segments after merge_short=%d (min_len=%.2fs)",
            len(speaker_segments),
            min_len,
        )
        
        # --- NEW: smooth short speaker "islands" (A-B-A型の短い島を吸収) ---
        max_island_sec = float(
            getattr(config.diarization, "max_island_sec", 2.0)
        )
        speaker_segments = smooth_speaker_islands(
            speaker_segments,
            max_island_sec=max_island_sec,
        )
        LOGGER.info(
            "[POST-ISLAND] segments after smoothing=%d (max_island_sec=%.2fs)",
            len(speaker_segments),
            max_island_sec,
        )

        # --- ALIGNMENT SANITY CHECK: ended too early? ---
        if speaker_segments and speaker_segments[-1]["end"] < duration_sec * 0.9:
            LOGGER.warning(
                f"[ALIGNER-FIX] alignment ended early at {speaker_segments[-1]['end']:.1f}s (<90% of audio). Expanding fallback."
            )
            speaker_segments = [{"start": 0.0, "end": duration_sec, "speaker": "SPEAKER_00"}]

    # --- POST-DIAR: 実効話者数の整理＆短命スピーカー吸収 ---
    if speaker_segments:
        speaker_segments = sanitize_speaker_timeline(
            speaker_segments,
            duration_sec=duration_sec,
            config=config.diarization,
        )
        LOGGER.info("[POST-DIAR] final_segments=%d", len(speaker_segments))

    # === DIAR stats: effective speakers from final timeline ===
    if speaker_segments:
        final_speakers = {seg.get("speaker", "SPEAKER_00") for seg in speaker_segments}
        num_speakers_effective = len(final_speakers)
    else:
        num_speakers_effective = 0

    add_metric("diarizer.num_speakers_effective", int(num_speakers_effective))

    min_speakers_cfg = int(getattr(config.diarization, "min_speakers", 1))
    min_unmet = bool(
        num_speakers_found >= min_speakers_cfg
        and num_speakers_effective < min_speakers_cfg
    )
    add_metric("diarizer.min_speakers_unmet", min_unmet)

    if min_unmet:
        LOGGER.warning(
            "[DIAR] min_speakers_unmet: min=%d, found=%d, effective=%d",
            min_speakers_cfg,
            num_speakers_found,
            num_speakers_effective,
        )
    else:
        LOGGER.info(
            "[DIAR] speakers: min=%d, found=%d, effective=%d",
            min_speakers_cfg,
            num_speakers_found,
            num_speakers_effective,
        )

    # NOTE: 方針Aではテキストは ASR の「生 words」のまま扱う
    add_metric("asr.word_count", len(words))
    
    if not speaker_segments and words:
        LOGGER.warning("DP alignment yielded no segments. Falling back to a single speaker segment.")
        valid_words = [
            w for w in words
            if w.get("start") is not None and w.get("end") is not None
        ]
        if valid_words:
            speaker_segments = [
                {
                    "start": float(valid_words[0]["start"]),
                    "end":   float(valid_words[-1]["end"]),
                    "speaker": "SPEAKER_00",
                }
            ]
        else:
            speaker_segments = []
        is_fallback = True

        # フォールバック後も一応 post-diar を通しておく（ほぼ no-op 想定）
        if speaker_segments:
            speaker_segments = sanitize_speaker_timeline(
                speaker_segments,
                duration_sec=duration_sec,
                config=config.diarization,
            )
            LOGGER.info("[POST-DIAR] final_segments(fallback)=%d", len(speaker_segments))

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
