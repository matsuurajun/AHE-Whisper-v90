# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, List, Any, Tuple

from ahe_whisper.config import AlignerConfig

class OverlapDPAligner:
    def __init__(self, config: AlignerConfig) -> None:
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.delta_switch = config.delta_switch
        self.non_speech_th = config.non_speech_th

    def align(
        self,
        word_info: List[Dict[str, Any]],
        vad_probs: np.ndarray,
        spk_probs: np.ndarray,
        grid_times: np.ndarray
    ) -> List[Tuple[float, float, int]]:
        
        # === DEBUG (AHE): Aligner input summary ===
        import inspect, sys
        print("[ALIGN-CALLER]", inspect.stack()[1].filename, inspect.stack()[1].lineno)
        print("[ALIGN-MODULE]", __name__, "loaded_from", sys.modules[__name__].__file__)
        
        try:
            total_words = len(word_info) if word_info is not None else 0
            print(f"[TRACE-ALIGNER-ENTRY] len(word_info)={total_words}, "
                  f"type={type(word_info)}, "
                  f"sample={word_info[:3] if isinstance(word_info, list) else 'N/A'}")
        except Exception as e:
            print(f"[TRACE-ALIGNER-ENTRY] word_info inspect failed: {e}")
        
        if not word_info or len(grid_times) == 0:
            return []

        num_frames = len(grid_times)
        num_speakers = spk_probs.shape[1]
        
        if num_speakers == 0:
            return []
        
        # === 改良版: 話者変化が検出できない場合の対策 ===
        # VAD確率を基に無音区間を検出し、そこでセグメントを分割
        
        cost = np.full((num_frames, num_speakers), np.inf, dtype=np.float32)
        path = np.full((num_frames, num_speakers), -1, dtype=np.int32)
        
        word_costs = self._precompute_word_costs(word_info, grid_times)
        
        # === 改良版: スコア最大化方針 (話者分離有効化) ===
        # 各スコアを「確率が高い方が良い」方向に符号反転し、最大化問題に変換する
        # 初期スコア
        for k in range(num_speakers):
            cost[0, k] = -(
                self.alpha * vad_probs[0] +
                self.beta  * spk_probs[0, k] -
                self.gamma * word_costs[0]
            )

        # DP更新（スコア最大化）
        for t in range(1, num_frames):
            vad_score = self.alpha * vad_probs[t]
            word_score_t = -self.gamma * word_costs[t]
            
            for k in range(num_speakers):
                spk_score = self.beta * spk_probs[t, k]
                base_score = vad_score + spk_score + word_score_t
                
                # 話者切替ペナルティをスコア方向で適用（切替時に少し不利）
                prev_scores = cost[t - 1, :] - self.delta_switch * (
                    1 - np.eye(num_speakers, dtype=np.float32)[k]
                )
                
                best_prev_k = np.argmax(prev_scores)
                max_prev_score = prev_scores[best_prev_k]
                
                cost[t, k] = max_prev_score + base_score
                path[t, k] = best_prev_k
                
        # バックトラッキング修正 (最大スコアを採用)
        final_path = np.zeros(num_frames, dtype=np.int32)
        if num_frames > 0:
            final_path[-1] = np.argmax(cost[-1, :])
            for t in range(num_frames - 2, -1, -1):
                final_path[t] = path[t + 1, final_path[t + 1]]

        # バックトラッキング
        final_path = np.zeros(num_frames, dtype=np.int32)
        if num_frames > 0:
            final_path[-1] = np.argmin(cost[-1, :])
            for t in range(num_frames - 2, -1, -1):
                final_path[t] = path[t + 1, final_path[t + 1]]

        # === 改良されたセグメント生成 ===
        segments = []
        if num_frames > 0:
            # 無音区間を検出（VAD確率が低い箇所）
            silence_threshold = 0.3  # VAD確率がこれ以下を無音とみなす
            min_silence_duration = 1.0  # 最小無音区間長（秒）
            max_segment_duration = 30.0  # 最大セグメント長（秒）
            
            # 無音区間の検出
            is_silence = vad_probs < silence_threshold
            silence_starts = []
            silence_ends = []
            
            in_silence = False
            silence_start_idx = 0
            
            for t in range(len(is_silence)):
                if is_silence[t] and not in_silence:
                    # 無音区間開始
                    silence_start_idx = t
                    in_silence = True
                elif not is_silence[t] and in_silence:
                    # 無音区間終了
                    silence_duration = grid_times[t] - grid_times[silence_start_idx]
                    if silence_duration >= min_silence_duration:
                        silence_starts.append(silence_start_idx)
                        silence_ends.append(t)
                    in_silence = False
            
            # セグメント生成
            segment_start_idx = 0
            segment_start_time = grid_times[0]
            current_speaker = final_path[0]
            
            print(f"[SEGMENT-DEBUG] Detected {len(silence_starts)} silence periods")
            
            # 話者変化点と無音区間を使ってセグメントを分割
            for t in range(1, num_frames):
                current_time = grid_times[t]
                segment_duration = current_time - segment_start_time
                
                # セグメント分割条件
                should_split = False
                split_reason = ""
                
                # 1. 話者が変わった
                if final_path[t] != current_speaker:
                    should_split = True
                    split_reason = "speaker_change"
                
                # 2. 無音区間の中央付近
                for i, (s_start, s_end) in enumerate(zip(silence_starts, silence_ends)):
                    if s_start <= t <= s_end and segment_duration > 5.0:  # 5秒以上のセグメント
                        should_split = True
                        split_reason = f"silence_{i}"
                        break
                
                # 3. セグメントが長すぎる
                if segment_duration >= max_segment_duration:
                    should_split = True
                    split_reason = "max_duration"
                
                if should_split:
                    # セグメントを追加
                    segments.append((segment_start_time, current_time, current_speaker))
                    print(f"[SEGMENT] {segment_start_time:.2f}-{current_time:.2f}s, "
                          f"speaker={current_speaker}, reason={split_reason}")
                    
                    # 次のセグメント開始
                    segment_start_time = current_time
                    if final_path[t] != current_speaker:
                        current_speaker = final_path[t]
                        segment_start_idx = t
            
            # 最後のセグメント
            end_time = grid_times[-1]
            segments.append((segment_start_time, end_time, current_speaker))
            print(f"[SEGMENT-FINAL] {segment_start_time:.2f}-{end_time:.2f}s, speaker={current_speaker}")
        
        # === フォールバック: セグメントが少なすぎる場合 ===
        if len(segments) <= 1 and num_frames > 0:
            print("[FALLBACK] Too few segments, creating artificial splits")
            segments = []
            segment_duration = 20.0  # 20秒ごとに分割
            
            # 話者ごとの平均確率を計算
            avg_spk_probs = np.mean(spk_probs, axis=0)
            dominant_speaker = np.argmax(avg_spk_probs)
            
            total_duration = grid_times[-1] - grid_times[0]
            num_segments = max(1, int(total_duration / segment_duration))
            
            for i in range(num_segments):
                start_time = grid_times[0] + i * segment_duration
                end_time = min(grid_times[-1], start_time + segment_duration)
                
                # この区間の話者を決定
                start_idx = np.searchsorted(grid_times, start_time)
                end_idx = np.searchsorted(grid_times, end_time)
                if end_idx > start_idx:
                    segment_path = final_path[start_idx:end_idx]
                    # 最頻値を話者とする
                    speaker = int(np.median(segment_path))
                else:
                    speaker = dominant_speaker
                
                segments.append((start_time, end_time, speaker))
                print(f"[FALLBACK-SEGMENT] {start_time:.2f}-{end_time:.2f}s, speaker={speaker}")
        
        print(f"[ALIGNER-RESULT] Generated {len(segments)} segments covering {segments[-1][1] if segments else 0:.1f}s")
        return segments

    def _precompute_word_costs(self, word_info: List[Dict[str, Any]], grid_times: np.ndarray) -> np.ndarray:
        word_costs = np.ones_like(grid_times, dtype=np.float32)
        if not word_info:
            return word_costs
        
        for word in word_info:
            w_start = word.get("start")
            w_end = word.get("end")
            
            # MLX-Whisper fallback: use 'confidence' or 'avg_logprob' if 'prob' missing
            prob_raw = word.get("prob")
            if prob_raw is None:
                prob_raw = word.get("confidence", None)
            if prob_raw is None:
                prob_raw = word.get("avg_logprob", None)
            if prob_raw is None:
                prob_raw = 0.8  # safe fallback
            
            try:
                w_start = float(w_start)
                w_end = float(w_end)
                prob = float(prob_raw)
            except (TypeError, ValueError):
                continue
            
            if w_end <= w_start:
                continue
            
            start_idx = np.searchsorted(grid_times, w_start, side="left")
            end_idx = np.searchsorted(grid_times, w_end, side="right")
            clamped_prob = max(0.0, min(1.0, prob))
            word_costs[start_idx:end_idx] = 1.0 - clamped_prob
            
        return word_costs
