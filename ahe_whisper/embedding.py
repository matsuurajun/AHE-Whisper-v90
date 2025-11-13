# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List
import logging

from ahe_whisper.utils import safe_l2_normalize
from ahe_whisper.features import Featurizer
from ahe_whisper.frontend_spec import load_spec_for_model, resolve_cmvn_policy
from ahe_whisper.config import EmbeddingConfig

LOGGER = logging.getLogger("ahe_whisper_worker")


def build_er2v2_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    if config.intra_threads is not None:
        sess_options.intra_op_num_threads = config.intra_threads
    sess_options.inter_op_num_threads = config.inter_threads

    providers = ["CPUExecutionProvider"]
    if config.prefer_coreml_ep and "CoreMLExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CoreMLExecutionProvider")

    return ort.InferenceSession(str(model_path), sess_options, providers=providers)


def warmup_er2v2(session: ort.InferenceSession) -> None:
    input_name = session.get_inputs()[0].name  # "x"
    dummy_input = np.random.randn(1, 200, 80).astype(np.float32)
    session.run(None, {input_name: dummy_input})


def er2v2_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    audio_chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig
) -> np.ndarray:
    
    # --- load frontend spec & CMVN ---
    spec, _ = load_spec_for_model(model_path)
    cmvn_policy = resolve_cmvn_policy(model_path.parent)
    featurizer = Featurizer(spec)

    input_name = session.get_inputs()[0].name    # == "x"
    output_name = session.get_outputs()[0].name  # == "embedding"
    emb_dim = config.embedding_dim               # == 192

    # --- Extract features ---
    features = [
        featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        for chunk in audio_chunks
    ]

    valid_indices = [
        i for i, f in enumerate(features)
        if f is not None and f.shape[0] > 0
    ]

    final_embeddings = np.zeros((len(audio_chunks), emb_dim), dtype=np.float32)

    if not valid_indices:
        return final_embeddings

    valid_features = [features[i] for i in valid_indices]
    feat_lens = [f.shape[0] for f in valid_features]

    order = np.argsort(feat_lens)
    buffer = np.zeros((len(valid_features), emb_dim), dtype=np.float32)

    batch_start = 0
    while batch_start < len(order):
        curr_len = feat_lens[order[batch_start]]
        bucket_max = ((curr_len // config.bucket_step) + 1) * config.bucket_step

        batch_end = batch_start
        while (
            batch_end < len(order)
            and feat_lens[order[batch_end]] <= bucket_max
            and (batch_end - batch_start) < config.batch_cap
        ):
            batch_end += 1

        idxs = order[batch_start:batch_end]
        max_len = feat_lens[idxs[-1]]

        batch_input = np.zeros(
            (len(idxs), max_len, 80), dtype=np.float32
        )

        for i, idx in enumerate(idxs):
            feat = valid_features[idx]
            batch_input[i, :feat.shape[0], :] = feat

        outputs = session.run([output_name], {input_name: batch_input})[0]

        for i, idx in enumerate(idxs):
            buffer[idx, :] = outputs[i]

        batch_start = batch_end

    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    # --- 重要：必ず L2 normalize ---
    return safe_l2_normalize(final_embeddings.astype(np.float32))
