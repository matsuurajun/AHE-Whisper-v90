# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort
from pathlib import Path
import logging
from typing import Any

LOGGER = logging.getLogger("ahe_whisper_worker")

class VAD:
    def __init__(self, model_path: Path, config: Any) -> None:
        self.session = ort.InferenceSession(str(model_path))
        self.config = config
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.uses_combined_state = 'state' in self.input_names
        self.reset_states()
        LOGGER.info(f"VAD interface detected: uses_combined_state={self.uses_combined_state}")

    def reset_states(self) -> None:
        state_shape = [2, 1, 128]
        try:
            if self.uses_combined_state:
                state_input = next((i for i in self.session.get_inputs() if i.name == 'state'), None)
                if state_input and all(isinstance(d, int) for d in state_input.shape):
                    state_shape = state_input.shape
            else:
                h_input = next((i for i in self.session.get_inputs() if i.name == 'h'), None)
                if h_input and all(isinstance(d, int) for d in h_input.shape):
                     state_shape = [2] + h_input.shape[1:]
        except Exception:
            LOGGER.warning("Could not determine VAD state shape from model, using default.")
        self._state = np.zeros(state_shape, dtype=np.float32)

    def __call__(self, x: np.ndarray, sr: int = 16000) -> np.ndarray:
        if len(x.shape) == 1: x = np.expand_dims(x, 0)
        
        sr_tensor = np.array(sr, dtype=np.int64)

        if self.uses_combined_state:
            ort_inputs = {'input': x, 'state': self._state, 'sr': sr_tensor}
            ort_outs = self.session.run(None, ort_inputs)
            out, self._state = ort_outs
        else:
            h, c = self._state[0:1], self._state[1:2]
            ort_inputs = {'input': x, 'h': h, 'c': c, 'sr': sr_tensor}
            ort_outs = self.session.run(None, ort_inputs)
            out, h_new, c_new = ort_outs
            self._state = np.concatenate([h_new, c_new], axis=0)

        return out.squeeze()

    def get_speech_probabilities(self, waveform: np.ndarray, sr: int, grid_hz: int):
        window_size = self.config.window_size_samples
        
        probs = []
        self.reset_states()
        
        for i in range(0, len(waveform), window_size):
            chunk = waveform[i: i + window_size]
            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)))
            prob = self(chunk.reshape(1, -1), sr)
            probs.append(float(prob))

        probs = np.asarray(probs, dtype=np.float32)
        
        total_sec = len(waveform) / sr
        if total_sec <= 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        
        num_grid_points = int(total_sec * grid_hz)
        if num_grid_points <= 0:
             return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        grid_times = np.linspace(0.0, total_sec, num_grid_points, endpoint=False, dtype=np.float32)

        frame_times = np.arange(len(probs), dtype=np.float32) * (window_size / sr)

        if frame_times.size < 2:
            fill_value = probs[0] if probs.size > 0 else 0.0
            interp_probs = np.full_like(grid_times, fill_value, dtype=np.float32)
        else:
            interp_probs = np.interp(grid_times, frame_times, probs).astype(np.float32)
        
        # === DEBUG: VAD probabilities summary ===
        try:
            print(f"[DEBUG-VAD] waveform_len={len(waveform)}, sr={sr}, grid_hz={grid_hz}")
            print(f"[DEBUG-VAD] probs.shape={probs.shape}, grid_times.shape={grid_times.shape}")
            if len(probs) > 0:
                above_th = np.sum(probs > 0.5)
                print(f"[DEBUG-VAD] proportion above 0.5 = {above_th / len(probs):.4f}")
            else:
                print("[DEBUG-VAD] No VAD probabilities computed.")
        except Exception as e:
            print(f"[DEBUG-VAD] Debug print failed: {e}")

        return interp_probs, grid_times
