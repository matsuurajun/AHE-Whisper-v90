# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Optional
import logging

from ahe_whisper.utils import safe_l2_normalize
from ahe_whisper.features import Featurizer
from ahe_whisper.frontend_spec import load_spec_for_model, resolve_cmvn_policy
from ahe_whisper.config import EmbeddingConfig

LOGGER = logging.getLogger("ahe_whisper_worker")

def build_ecapa_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    if config.intra_threads is not None:
        sess_options.intra_op_num_threads = config.intra_threads
    sess_options.inter_op_num_threads = config.inter_threads
    
    providers = ['CPUExecutionProvider']
    if config.prefer_coreml_ep and 'CoreMLExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CoreMLExecutionProvider')
        
    return ort.InferenceSession(str(model_path), sess_options, providers=providers)

def warmup_ecapa(session: ort.InferenceSession) -> None:
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 200, 80).astype(np.float32)
    session.run(None, {input_name: dummy_input})

def ecapa_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    audio_chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig
) -> np.ndarray:
    
    spec, _ = load_spec_for_model(model_path)
    cmvn_policy = resolve_cmvn_policy(model_path.parent)
    featurizer = Featurizer(spec)
    input_name = session.get_inputs()[0].name
    
    features = [featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy) for chunk in audio_chunks]
    
    valid_indices = [i for i, f in enumerate(features) if f is not None and f.shape[0] > 0]
    final_embeddings = np.zeros((len(audio_chunks), config.embedding_dim), dtype=np.float32)

    if not valid_indices:
        return final_embeddings
        
    valid_features = [features[i] for i in valid_indices]
    
    feature_lengths = [f.shape[0] for f in valid_features]
    sorted_indices_of_valid = np.argsort(feature_lengths)
    
    results_buffer = np.zeros((len(valid_features), config.embedding_dim), dtype=np.float32)
    
    batch_start = 0
    while batch_start < len(sorted_indices_of_valid):
        current_len = feature_lengths[sorted_indices_of_valid[batch_start]]
        bucket_max_len = ((current_len // config.bucket_step) + 1) * config.bucket_step
        
        batch_end = batch_start
        while (batch_end < len(sorted_indices_of_valid) and 
               feature_lengths[sorted_indices_of_valid[batch_end]] <= bucket_max_len and 
               (batch_end - batch_start) < config.batch_cap):
            batch_end += 1
        
        batch_indices_in_sorted_valid = sorted_indices_of_valid[batch_start:batch_end]
        
        max_len_in_batch = feature_lengths[batch_indices_in_sorted_valid[-1]]
        
        batch_input = np.zeros((len(batch_indices_in_sorted_valid), max_len_in_batch, 80), dtype=np.float32)
        for i, idx_in_valid in enumerate(batch_indices_in_sorted_valid):
            feat = valid_features[idx_in_valid]
            batch_input[i, :feat.shape[0], :] = feat
        
        batch_embs = session.run(None, {input_name: batch_input})[0]
        
        for i, idx_in_valid in enumerate(batch_indices_in_sorted_valid):
            results_buffer[idx_in_valid, :] = batch_embs[i]
            
        batch_start = batch_end

    for i, original_chunk_idx in enumerate(valid_indices):
        final_embeddings[original_chunk_idx, :] = results_buffer[i]
        
    return safe_l2_normalize(final_embeddings)
