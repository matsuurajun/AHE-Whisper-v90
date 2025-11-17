# -*- coding: utf-8 -*-
import sys
import os
import logging
import time
import traceback
import json
import io
from pathlib import Path
from multiprocessing import Queue
from typing import Dict, Any, List, Tuple, Optional

from dacite import from_dict, Config as DaciteConfig

from ahe_whisper.config import AppConfig
from ahe_whisper.pipeline import run as run_ai_pipeline
from ahe_whisper.exporter import Exporter
from ahe_whisper.utils import reset_metrics

def _merge_continuous_speaker_segments(segments: List, max_gap: float, min_dur: float) -> List:
    if not segments:
        return []

    # --- sort safely for both dict and tuple structures ---
    if isinstance(segments[0], dict):
        segments.sort(key=lambda x: float(x.get("start", 0.0)))
    else:
        segments.sort(key=lambda x: x[0])

    merged = []
    if isinstance(segments[0], dict):
        cs, ce, ck = float(segments[0]["start"]), float(segments[0]["end"]), segments[0].get("speaker", "SPEAKER_00")
        for seg in segments[1:]:
            ns, ne, nk = float(seg["start"]), float(seg["end"]), seg.get("speaker", "SPEAKER_00")
            if nk == ck and (ns - ce) <= max_gap:
                ce = max(ce, ne)
            else:
                if ce - cs >= min_dur:
                    merged.append({"start": float(cs), "end": float(ce), "speaker": ck})
                cs, ce, ck = ns, ne, nk
        if ce - cs >= min_dur:
            merged.append({"start": float(cs), "end": float(ce), "speaker": ck})
    else:
        cs, ce, ck = segments[0]
        for ns, ne, nk in segments[1:]:
            if nk == ck and (ns - ce) <= max_gap:
                ce = max(ce, ne)
            else:
                if ce - cs >= min_dur:
                    merged.append((cs, ce, ck))
                cs, ce, ck = ns, ne, nk
        if ce - cs >= min_dur:
            merged.append((cs, ce, ck))

    return merged

def worker_process_loop(job_q: Queue, result_q: Queue, log_q: Queue, project_root_str: str) -> None:
    project_root = Path(project_root_str)
    
    # === [AHE PATCH] 全GUIログをバッファにも保存する設定 ===
    import io
    LOG_BUFFER = io.StringIO()

    class BufferHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = self.format(record)
            LOG_BUFFER.write(msg + "\n")

    class QueueHandler(logging.Handler):
        def __init__(self, q: Queue) -> None:
            super().__init__()
            self.q = q
            self._ahe_queue_handler = True
        def emit(self, record: logging.LogRecord) -> None:
            self.q.put(self.format(record))

    class _LogRedirect:
        def __init__(self, logger: logging.Logger, level: int, orig_fd: Optional[int]) -> None:
            self._logger, self._level, self._orig_fd = logger, level, orig_fd
        def write(self, msg: str) -> None:
            line = (msg or "").strip()
            if not line: return
            if "Fetching" in line or "Downloading" in line or "%|" in line: self._logger.info(line)
            elif 'warning' in line.lower(): self._logger.warning(line)
            else: self._logger.log(self._level, line)
        def flush(self) -> None: pass
        def isatty(self) -> bool: return False
        def fileno(self) -> int:
            if self._orig_fd is not None: return self._orig_fd
            raise io.UnsupportedOperation("fileno")

    logger = logging.getLogger("ahe_whisper_worker")
    handler = QueueHandler(log_q)
    handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(module)s: %(message)s', datefmt='%H:%M:%S'))
    
    if not any(getattr(h, "_ahe_queue_handler", False) for h in logger.handlers):
        logger.addHandler(handler)
    
    logger.propagate = False
    logger.setLevel(logging.INFO)
    
    # === [AHE PATCH] バッファハンドラも同時に登録 ===
    buffer_handler = BufferHandler()
    buffer_handler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s-%(module)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(buffer_handler)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    
    orig_stdout_fd: Optional[int] = None
    try:
        if hasattr(sys, '__stdout__') and sys.__stdout__ is not None and hasattr(sys.__stdout__, 'fileno'):
            orig_stdout_fd = sys.__stdout__.fileno()
    except (AttributeError, io.UnsupportedOperation):
        pass

    sys.stdout = _LogRedirect(logger, logging.INFO, orig_stdout_fd)
    sys.stderr = _LogRedirect(logger, logging.ERROR, orig_stdout_fd)

    dacite_config = DaciteConfig(type_hooks={int: lambda data: int(round(data)) if isinstance(data, float) else data})

    def generate_rich_perf_report(step_times: List[Tuple[str, float]], result: Dict[str, Any]) -> List[str]:
        total_time = sum(t for _, t in step_times) or 1.0
        duration = result.get('duration_sec', 0.0)
        rtf = total_time / max(duration, 1.0)
        
        metrics = result.get('metrics', {})
        asr_coverage = metrics.get('asr.coverage_ratio', 0.0) * 100
        vad_speech = metrics.get('vad.speech_ratio', 0.0) * 100

        # diarizer-related metrics
        num_found = metrics.get("diarizer.num_speakers_found")
        num_eff = metrics.get("diarizer.num_speakers_effective")
        min_unmet = bool(metrics.get("diarizer.min_speakers_unmet", False))
        cluster_mass = metrics.get("diarizer.cluster_mass", None)

        report = [f"--- Performance Report (RTF={rtf:.3f}) ---"]
        report.append(f"- ASR Coverage             : {asr_coverage:6.2f}%")
        report.append(f"- VAD Speech Ratio         : {vad_speech:6.2f}%")

        if num_found is not None and num_eff is not None:
            line = f"- Speakers (found/effective): {int(num_found):3d} / {int(num_eff):3d}"
            if min_unmet:
                line += "  [MIN_UNMET]"
            report.append(line)

        if cluster_mass:
            try:
                mass_str = ", ".join(f"{float(m):.3f}" for m in cluster_mass)
            except Exception:
                mass_str = str(cluster_mass)
            report.append(f"- Cluster mass fractions   : [{mass_str}]")

        report.append("-" * 38)
        for name, step_time in step_times:
            report.append(f"- {name:<25}: {step_time:>7.2f}s ({(step_time / total_time * 100):.1f}%)")
        return report

    try:
        while True:
            # === [AHE PATCH] 複数ジョブ実行時にバッファを毎回リセット ===
            LOG_BUFFER.seek(0)
            LOG_BUFFER.truncate(0)

            job = job_q.get()
            if job is None: 
                logger.info("Shutdown signal received.")
                break
            try:
                config_dict, audio_path = job
                config = from_dict(data_class=AppConfig, data=config_dict, config=dacite_config)
                
                logger.info(f"--- Job Start: {Path(audio_path).name} ---")
                reset_metrics()
                step_times, t_start = [], time.perf_counter()
                #res = run_ai_pipeline(audio_path, config, project_root)
                #step_times.append(("Core AI Pipeline", time.perf_counter() - t_start))
                # --- Stage-by-stage measurement ---
                step_times = []
                t_total_start = time.perf_counter()
                
                # --- ASR ---
                t_asr = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="asr")
                step_times.append(("ASR (mlx_whisper)", time.perf_counter() - t_asr))
                
                # --- VAD ---
                t_vad = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="vad")
                step_times.append(("VAD (silero_vad)", time.perf_counter() - t_vad))
                
                # --- DIAR ---
                t_diar = time.perf_counter()
                run_ai_pipeline(audio_path, config, project_root, stage="diar")
                step_times.append(("DIAR (wespeaker)", time.perf_counter() - t_diar))
                
                # --- FULL PIPELINE (ALIGN + EXPORT) ---
                t_main = time.perf_counter()
                res = run_ai_pipeline(audio_path, config, project_root)
                step_times.append(("Core AI Pipeline (Align+Post)", time.perf_counter() - t_main))
                
                # --- ensure exportable structure ---
                if "speaker_segments" not in res or not res["speaker_segments"]:
                    logger.warning("[PIPELINE-WORKER] No speaker segments found. Creating fallback segment for export.")
                    dur = res.get("duration_sec", 0.0)
                    res["speaker_segments"] = [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]
                
                # reconstruct text if missing
                if "text" not in res or not isinstance(res["text"], str) or not res["text"].strip():
                    if "words" in res:
                        res["text"] = " ".join([w.get("word", "") for w in res["words"] if w.get("word")])
                        logger.info(f"[PIPELINE-WORKER] Reconstructed text length: {len(res['text'])}")
                    else:
                        res["text"] = ""

                t_post_start = time.perf_counter()
                is_fallback = res.get("is_fallback", False)
                min_duration = config.diarization.min_fallback_duration_sec if is_fallback else config.diarization.min_speaker_duration_sec
                merged = _merge_continuous_speaker_segments(res.get("speaker_segments", []), config.diarization.max_merge_gap_sec, min_duration)
                res["speaker_segments"] = merged
                step_times.append(("Post-processing", time.perf_counter() - t_post_start))
                
                # --- attach text to each segment for export ---
                if "speaker_segments" in res and "words" in res:
                    for seg in res["speaker_segments"]:
                        seg_words = [
                            w.get("word", w.get("text", ""))  # 安全に取得
                            for w in res.get("words", [])
                            if isinstance(w, dict) and seg["start"] <= w.get("start", 0) < seg["end"]
                        ]
                        seg["text"] = "".join(seg_words).strip()
                    valid_segs = [s for s in res["speaker_segments"] if s.get("text")]
                    logger.info(f"[PIPELINE-WORKER] Segments with text: {len(valid_segs)} / {len(res['speaker_segments'])}")
                    
                    # --- ensure backward-compatible format for exporter ---
                    if "speaker_segments" in res and isinstance(res["speaker_segments"], list):
                        segs = []
                        for seg in res["speaker_segments"]:
                            start = seg.get("start", 0.0)
                            end = seg.get("end", start)
                            spk = seg.get("speaker", 0)
                            segs.append((start, end, spk))
                        res["speaker_segments_raw"] = segs
                        logger.info(f"[PIPELINE-WORKER] Exporter-compatible speaker_segs_raw: {len(segs)} items")

                t_export_start = time.perf_counter()
                exporter = Exporter(config.output_dir, config.export)
                run_dir = exporter.save(res, Path(audio_path).stem)
                (run_dir / "run_config.json").write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
                step_times.append(("Exporting Files", time.perf_counter() - t_export_start))

                report = generate_rich_perf_report(step_times, res)
                logger.info("\n" + "\n".join(report))
                (run_dir / "performance.log").write_text("\n".join(report), encoding="utf-8")
                
                # === [AHE PATCH] ジョブ全体のGUIログを performance.log に追記 ===
                full_log_path = run_dir / "performance.log"
                with open(full_log_path, "a", encoding="utf-8") as f:
                    f.write("\n\n--- Full GUI Log ---\n")
                    f.write(LOG_BUFFER.getvalue())
                logger.info(f"[PIPELINE] Full GUI log appended to {full_log_path}")

                result_q.put({"success": True, "output_dir": str(run_dir)})
            except Exception as e:
                logger.error(f"Critical error in worker: {e}\n{traceback.format_exc()}")
                result_q.put({"success": False, "error": str(e)})
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
