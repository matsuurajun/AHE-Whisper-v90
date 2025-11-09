# -*- coding: utf-8 -*-
import json
import traceback
import numpy as np
import textwrap
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ahe_whisper.config import ExportConfig
from ahe_whisper.logger import LOGGER

_CJK_RE = re.compile(r'[\u3000-\u303F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]')

def _wrap_cjk(text: str, width: int) -> str:
    if width <= 0 or not text: return text
    lines, current_line, current_width = [], "", 0
    for char in text:
        char_width = 2 if _CJK_RE.match(char) else 1
        if current_width + char_width > width:
            lines.append(current_line)
            current_line, current_width = char, char_width
        else:
            current_line += char
            current_width += char_width
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)

def _tag_words_with_speaker(words: List[Dict[str, Any]], speaker_segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not speaker_segs:
        for word in words:
            word['speaker'] = None
        return words
    
    segs = sorted(speaker_segs, key=lambda s: s['start'])
    seg_starts = [s['start'] for s in segs]
    
    EPS = 1e-6
    
    for word in words:
        start_raw = word.get('start')
        end_raw = word.get('end')
        
        if start_raw is None or end_raw is None:
            word['speaker'] = None
            continue
        
        try:
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):
            word['speaker'] = None
            continue
        
        if end <= start:
            word['speaker'] = None
            continue
        
        word_mid = (start + end) / 2.0
        idx = int(np.searchsorted(seg_starts, word_mid, side='right')) - 1
        
        if 0 <= idx < len(segs):
            seg = segs[idx]
            if (seg['start'] - EPS) <= word_mid <= (seg['end'] + EPS):
                word['speaker'] = seg['speaker']
            else:
                word['speaker'] = None
        else:
            word['speaker'] = None
            
    return words

class Exporter:
    def __init__(self, output_dir: str, config: ExportConfig) -> None:
        self.output_dir = Path(output_dir)
        self.config = config

    def _format_time(self, seconds_total: float) -> str:
        try:
            sec = max(0.0, float(seconds_total))
            td = timedelta(seconds=sec)
        except (TypeError, ValueError):
            return "00:00:00,000"
            
        mm, ss = divmod(td.seconds, 60)
        hh, mm = divmod(mm, 60)
        hh += td.days * 24
        ms = td.microseconds // 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"
    
    def _default_serializer(self, o: Any) -> Any:
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.bool_): return bool(o)
        return str(o)
        
    def _wrap_text(self, text: str) -> str:
        width = self.config.srt_max_line_width
        if width <= 0: return text
        if _CJK_RE.search(text):
            return _wrap_cjk(text, width)
        return '\n'.join(textwrap.wrap(text, width=width, replace_whitespace=False, break_long_words=False, break_on_hyphens=False))

    def save(self, result: Dict[str, Any], basename: str) -> Path:
        
        LOGGER.info(f"[DEBUG-EXPORT] save() called, keys={list(result.keys())}")
        LOGGER.info(f"[DEBUG-EXPORT] output_dir={self.output_dir}, exists={self.output_dir.exists()}")
        
        safe_basename = re.sub(r'[^\w.\-]+', '_', basename).strip('._') or "run"
            
        run_dir = None
        for i in range(100):
            suffix = "" if i == 0 else f"-{i+1}"
            run_dir_name = f"{safe_basename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
            candidate_dir = self.output_dir / run_dir_name
            try:
                candidate_dir.mkdir(parents=True, exist_ok=False)
                run_dir = candidate_dir
                break
            except FileExistsError:
                continue
        
        if run_dir is None:
            raise IOError("Could not create a unique output directory after 100 attempts.")
        
        error_log_path = run_dir / "export_error.log"
        words, speaker_segs_raw = result.get('words', []), result.get('speaker_segments', [])
        
        final_segs = []
        if speaker_segs_raw:
            for seg in speaker_segs_raw:
                if isinstance(seg, (list, tuple)) and len(seg) >= 3:
                    start, end, spk_idx = seg[:3]  # 最初の3つだけを取り出す
                elif isinstance(seg, dict):
                    start = seg.get("start", 0.0)
                    end = seg.get("end", start)
                    spk_idx = seg.get("speaker", 0)
                else:
                    continue
                                
                try:
                    start = float(start)
                    end = float(end)
                except (TypeError, ValueError):
                    continue  # skip invalid segment
                
                seg_words: List[Dict[str, Any]] = []
                for w in words:
                    w_start_raw = w.get('start')
                    w_end_raw = w.get('end')
                    
                    if w_start_raw is None or w_end_raw is None:
                        continue
                    
                    try:
                        w_start = float(w_start_raw)
                        w_end = float(w_end_raw)
                    except (TypeError, ValueError):
                        continue
                    
                    if w_end <= w_start:
                        continue
                    
                    # --- 型安全比較 ---
                    if max(0.0, min(end, w_end) - max(start, w_start)) > 0.0:
                        seg_words.append(w)
                
                def _wtext(w):
                    return (w.get('text') or w.get('word') or '').strip()
                
                if text := " ".join(_wtext(w) for w in seg_words).strip():
                    if isinstance(spk_idx, str) and spk_idx.startswith("SPEAKER_"):
                        speaker_label = spk_idx
                    else:
                        try:
                            speaker_label = f"SPEAKER_{int(spk_idx):02d}"
                        except Exception:
                            speaker_label = "SPEAKER_00"
                    final_segs.append({
                        "start": float(start),
                        "end": float(end),
                        "speaker": speaker_label,
                        "text": text
                    })
        
        if words and final_segs:
            valid_words = [w for w in words if w.get("start") is not None and w.get("end") is not None]
            if valid_words:
                w_min = float(min(w["start"] for w in valid_words))
                w_max = float(max(w["end"] for w in valid_words))
                clamped_segs = []
                for seg in final_segs:
                    s = max(float(seg["start"]), w_min)
                    e = min(float(seg["end"]), w_max)
                    if e > s:
                        seg["start"], seg["end"] = s, e
                        clamped_segs.append(seg)
                final_segs = clamped_segs

        tagged_words = _tag_words_with_speaker(words, final_segs)
        result['words'] = tagged_words
        result['speaker_segments'] = final_segs

        try:
            raw_path = run_dir / f"{safe_basename}_raw_output.json"
            with open(raw_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=self._default_serializer)
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"Raw JSON export failed: {e}\n{traceback.format_exc()}\n")

        LOGGER.info(f"[DEBUG-EXPORT] final_segs count={len(final_segs)}")

        if not final_segs: return run_dir

        try:
            if "txt" in self.config.output_formats:
                txt_lines = [f"[{self._format_time(s['start'])}] {s['speaker']}: {s['text']}" for s in final_segs]
                (run_dir / f"{safe_basename}.txt").write_text("\n".join(txt_lines), 'utf-8')
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"TXT export failed: {e}\n{traceback.format_exc()}\n")
            
        try:
            if "srt" in self.config.output_formats:
                srt_lines = [f"{i+1}\n{self._format_time(s['start'])} --> {self._format_time(s['end'])}\n{s['speaker']}: {self._wrap_text(s['text'])}" for i, s in enumerate(final_segs)]
                (run_dir / f"{safe_basename}.srt").write_text("\n\n".join(srt_lines), 'utf-8')
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"SRT export failed: {e}\n{traceback.format_exc()}\n")
            
        try:
            if "json" in self.config.output_formats:
                json_output = {"segments": final_segs, "words": tagged_words, "metrics": result.get("metrics", {})}
                with open(run_dir / f"{safe_basename}.json", 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=2, ensure_ascii=False, default=self._default_serializer)
        except Exception as e:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"Final JSON export failed: {e}\n{traceback.format_exc()}\n")
        
        return run_dir
