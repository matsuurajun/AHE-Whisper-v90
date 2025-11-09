# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
from typing import Tuple

from ahe_whisper.utils import safe_softmax, safe_l2_normalize
from ahe_whisper.config import DiarizationConfig

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
            valid_similarities = np.tanh(valid_similarities * 2.5)
            print(f"[DEBUG-DIAR] AFTER norm: std={np.std(valid_similarities):.4f}, "
                  f"min={np.min(valid_similarities):.3f}, max={np.max(valid_similarities):.3f}")

        for k in range(num_speakers):
            interp_func = interp1d(valid_times, valid_similarities[:, k], kind='linear', bounds_error=False, fill_value="extrapolate")
            spk_probs[:, k] = interp_func(grid_times)
        
        # --- diagnostics ---
        self.last_valid_sims = valid_similarities.copy()
        
        return safe_softmax(spk_probs, tau=0.4 if np.std(valid_similarities) < 0.05 else 1.0)
