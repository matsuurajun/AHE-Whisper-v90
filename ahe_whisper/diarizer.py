# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from typing import Tuple
import logging

from ahe_whisper.utils import safe_softmax, safe_l2_normalize
from ahe_whisper.config import DiarizationConfig

LOGGER = logging.getLogger(__name__)
RNG = np.random.default_rng(42)

class Diarizer:
    def __init__(self, config: DiarizationConfig) -> None:
        self.config = config

    def cluster(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # [PATCH v90.90] CRITICAL ATTRIBUTE/INDEX ERROR FIX:
        # Safely handle empty embeddings array without referencing self.config.
        if embeddings.shape[0] == 0:
            embedding_dim = embeddings.shape[1] if (embeddings.ndim > 1 and embeddings.shape[1] > 0) else 192
            zero_centroid = np.zeros((1, embedding_dim), dtype=np.float32)
            return safe_l2_normalize(zero_centroid), np.array([], dtype=int)
            
        if embeddings.shape[0] < self.config.min_speakers:
            labels = np.zeros(embeddings.shape[0], dtype=int)
            centroids = np.mean(embeddings, axis=0, keepdims=True)
            return safe_l2_normalize(centroids), labels

        k_min = self.config.min_speakers
        k_max = min(self.config.max_speakers, embeddings.shape[0])
        
        k = max(k_min, min(k_max, int(np.sqrt(embeddings.shape[0]/2)) if embeddings.shape[0] > 8 else 2))

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(embeddings)
        
        attractors = safe_l2_normalize(kmeans.cluster_centers_)
        
        responsibilities = np.full((embeddings.shape[0], k), 1.0 / k, dtype=np.float32)

        for tau in self.config.em_tau_schedule:
            similarities = embeddings @ attractors.T
            responsibilities = safe_softmax(similarities, tau)
            
            new_attractors = np.zeros_like(attractors)
            for i in range(k):
                r_i = responsibilities[:, i]
                if np.sum(r_i) > 1e-6:
                    weighted_sum = np.sum(embeddings * r_i[:, np.newaxis], axis=0)
                    new_attractors[i] = safe_l2_normalize(weighted_sum)
                else:
                    random_idx = RNG.choice(embeddings.shape[0])
                    new_attractors[i] = safe_l2_normalize(embeddings[random_idx])

            attractors = new_attractors
        
        final_labels = np.argmax(responsibilities, axis=1)
        return attractors, final_labels

    def get_speaker_probabilities(
        self,
        embeddings: np.ndarray,
        valid_mask: np.ndarray,
        centroids: np.ndarray,
        grid_times: np.ndarray,
        hop_len: int,
        sr: int
    ) -> np.ndarray:
        
        # === DEBUG (AHE): diarizer input sanity ===
        try:
            print(f"[DEBUG-DIAR] grid_len={len(grid_times)}, "
                  f"embeddings.shape={getattr(embeddings, 'shape', None)}, "
                  f"valid_mask.sum={int(valid_mask.sum()) if valid_mask is not None else 'N/A'}, "
                  f"centroids.shape={getattr(centroids, 'shape', None)}, "
                  f"hop_len={hop_len}, sr={sr}")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug header failed: {e}")
        
        num_speakers = len(centroids)
        embedding_times = np.arange(len(embeddings)) * (hop_len / sr)
        
        similarities = embeddings @ centroids.T
        
        spk_probs = np.zeros((len(grid_times), num_speakers), dtype=np.float32)
        
        valid_times = embedding_times[valid_mask]
        valid_similarities = similarities[valid_mask, :]
        
        # === DEBUG (AHE): timeline & similarities ===
        try:
            print(f"[DEBUG-DIAR] valid_times_len={len(valid_times)}, "
                  f"valid_sims.shape={getattr(valid_similarities, 'shape', None)}")
            if len(valid_times) >= 2:
                print(f"[DEBUG-DIAR] valid_times.head="
                      f"{[round(float(t),2) for t in valid_times[:5]]} ...")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug valid* failed: {e}")
        
        # === DEBUG (AHE): early-exit condition check ===
        try:
            if len(valid_times) < 2:
                print(f"[DEBUG-DIAR] early-exit: len(valid_times)={len(valid_times)} -> "
                      f"{'copy-single or zeros then softmax' if len(valid_times)==1 else 'zeros then softmax'}")
        except Exception as e:
            print(f"[DEBUG-DIAR] debug early-exit failed: {e}")

        if len(valid_times) < 2:
            if len(valid_times) == 1:
                spk_probs[:, :] = valid_similarities[0, :]
            return safe_softmax(spk_probs)
        
        # === [PATCH v90.91-CONTRAST] Apply contrast normalization before interpolation ===
        var = np.std(valid_similarities)
        if var < 0.05:
            print(f"[DEBUG-DIAR] low variance (std={var:.4f}) → applying contrast normalization (pre-interp)")
            # Z-score normalization per frame
            valid_similarities = (valid_similarities - np.mean(valid_similarities, axis=1, keepdims=True)) / \
                                 (np.std(valid_similarities, axis=1, keepdims=True) + 1e-6)
            # Moderate contrast enhancement
            valid_similarities = np.tanh(valid_similarities * 3.0)
            print(f"[DEBUG-DIAR] AFTER norm: std={np.std(valid_similarities):.4f}, "
                  f"min={np.min(valid_similarities):.3f}, max={np.max(valid_similarities):.3f}")
            
            # === [v90.96][DIAR] Local-τ + Sharp + EMA + Centroid-Inertia ===
            # 前提：valid_similarities は shape=(T, K) の相関/コサイン類似度。tanh(*3) 適用済み。
            T, K = valid_similarities.shape
            
            # 1) Local std から τ(t) を作る（短窓で分離コントラストを推定）
            #    stdが低い→τを高めに（過剰分割抑制）、stdが高い→τを低めに（決定力UP）
            win = 41  # ≈0.8s @50Hz（あなたのgrid_hz=50前提）
            if win % 2 == 0:
                win += 1
            half = win // 2
            
            # valid_similarities を時間方向に std 計算（Kスピーカ平均の振幅で）
            loc_std = np.empty(T, dtype=np.float32)
            pad = np.pad(valid_similarities, ((half, half), (0, 0)), mode="edge")
            for t in range(T):
                sl = pad[t:t+win]
                loc_std[t] = np.std(sl, axis=1).mean()
            
            # τを std のレンジに応じて線形マッピング（要件に応じて調整可能）
            #   std=0.05 → τ=0.65（平坦→過分割しがちなので温度高め＝soft化）
            #   std=0.20 → τ=0.35（コントラスト十分→温度低め＝決定力）
            tau_min, tau_max = 0.35, 0.65
            std_lo, std_hi = 0.05, 0.20
            tau_t = tau_max - (np.clip(loc_std, std_lo, std_hi) - std_lo) * (tau_max - tau_min) / (std_hi - std_lo)
            tau_t = tau_t.astype(np.float32)  # shape=(T,)
            
            # 2) 温度付き softmax（時間tごとに τ(t) を適用）
            #    logits = valid_similarities / τ(t)
            logits = valid_similarities / tau_t[:, None]
            logits = logits - logits.max(axis=1, keepdims=True)  # 数値安定化
            spk_probs = np.exp(logits)
            spk_probs /= np.sum(spk_probs, axis=1, keepdims=True) + 1e-12  # shape=(T, K)
            
            # 3) 事後シャープ（弱ピークを強調しつつ、過剰分割はEMAで抑える）
            gamma = 1.30
            spk_probs = np.power(spk_probs, gamma)
            spk_probs /= np.sum(spk_probs, axis=1, keepdims=True) + 1e-12
            
            # 4) EMA 平滑（過剰なスイッチを抑制）
            #    alpha を小さくすると切替に粘り。音源によって 0.15〜0.30 を推奨。
            alpha = 0.22
            ema = np.empty_like(spk_probs)
            ema[0] = spk_probs[0]
            for t in range(1, T):
                ema[t] = alpha * spk_probs[t] + (1.0 - alpha) * ema[t-1]
            spk_probs = ema
            
            # 5) セントロイド慣性（前回重心とブレンドして attractor を安定化）
            #    既存の centroids: shape=(K, D) がある前提。無ければスキップ。
            try:
                if centroids is not None:
                    inertia = 0.70  # 0.6〜0.8推奨：大きいほど前回重心を保持して安定
                    if not hasattr(self, "_prev_centroids") or self._prev_centroids is None or \
                       getattr(self, "_prev_centroids", None) is None or \
                       getattr(self, "_prev_centroids", None) is False:
                        self._prev_centroids = centroids.copy()
                    else:
                        centroids = inertia * self._prev_centroids + (1.0 - inertia) * centroids
                        self._prev_centroids = centroids.copy()
            except NameError:
                pass  # centroids がスコープ外なら黙って無視

            # 6) 診断ログ（あなたの既存ログと整合）
            LOGGER.info("[DEBUG-DIAR-TAU] Local τ: min=%.2f, max=%.2f, mean=%.2f, std(valid_sims)=%.4f",
                        float(tau_t.min()), float(tau_t.max()), float(tau_t.mean()),
                        float(valid_similarities.std()))
            mm = float(np.mean(np.max(spk_probs, axis=1)))
            ent = float(-np.mean(np.sum(spk_probs * np.log(np.clip(spk_probs, 1e-9, 1.0)), axis=1)))
            LOGGER.info("[SPK-PROBS] (post) mean_max=%.3f, mean_entropy=%.3f", mm, ent)

            # 7) デバッグ支援（必要なら保存→即解放）
            setattr(self, "last_probs", spk_probs)
            LOGGER.info("[DEBUG-DIAR] last_probs available: shape=%s, mean_max=%.3f, entropy=%.3f",
                        str(spk_probs.shape), mm, ent)
            try:
                del self.last_probs
                LOGGER.debug("[DEBUG-DIAR] last_probs deleted to reduce memory footprint")
            except Exception as e:
                LOGGER.warning(f"[DEBUG-DIAR] could not delete last_probs: {e}")

            # === /[v90.96][DIAR] ===
            
            # === [v90.96-FIX] interpolate Local-τ probs to grid_times ===
            interp_out = np.zeros((len(grid_times), K), dtype=np.float32)
            for k in range(K):
                interp_func = interp1d(valid_times, spk_probs[:, k], kind='linear', bounds_error=False, fill_value="extrapolate")
                interp_out[:, k] = interp_func(grid_times)
            spk_probs = interp_out  # replace with full-resolution probabilities

        for k in range(num_speakers):
            interp_func = interp1d(valid_times, valid_similarities[:, k], kind='linear', bounds_error=False, fill_value="extrapolate")
            spk_probs[:, k] = interp_func(grid_times)
        
        # --- diagnostics ---
        self.last_valid_sims = valid_similarities.copy()

