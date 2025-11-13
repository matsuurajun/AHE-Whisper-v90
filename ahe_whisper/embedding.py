# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from ahe_whisper.utils import safe_l2_normalize
from ahe_whisper.features import Featurizer
from ahe_whisper.frontend_spec import load_spec_for_model, resolve_cmvn_policy
from ahe_whisper.config import EmbeddingConfig

LOGGER = logging.getLogger("ahe_whisper_worker")


def build_er2v2_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    ERes2NetV2 用の ONNXRuntime セッションを構築する。
    """
    sess_options = ort.SessionOptions()
    if config.intra_threads is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if config.inter_threads is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if config.prefer_coreml_ep and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[ER2V2] Building InferenceSession: model=%s, providers=%s, intra=%s, inter=%s",
        str(model_path),
        providers,
        getattr(config, "intra_threads", None),
        getattr(config, "inter_threads", None),
    )
    session = ort.InferenceSession(str(model_path), sess_options, providers=providers)

    # 簡易ログ（入出力の確認）
    try:
        in0 = session.get_inputs()[0]
        out0 = session.get_outputs()[0]
        LOGGER.info(
            "[ER2V2] IO signature: input(name=%s, shape=%s), output(name=%s, shape=%s)",
            in0.name,
            in0.shape,
            out0.name,
            out0.shape,
        )
    except Exception as e:
        LOGGER.warning("[ER2V2] Failed to inspect IO signature: %s", e)

    return session


def warmup_er2v2(session: ort.InferenceSession) -> None:
    """
    ER2V2 のウォームアップ。短いダミー入力で一度 forward しておく。
    """
    try:
        input_name = session.get_inputs()[0].name
    except Exception as e:
        LOGGER.warning("[ER2V2] warmup: failed to get input name: %s", e)
        return

    dummy = np.random.randn(1, 200, 80).astype(np.float32)
    try:
        _ = session.run(None, {input_name: dummy})
        LOGGER.info("[ER2V2] Warmup done: dummy shape=%s", dummy.shape)
    except Exception as e:
        LOGGER.warning("[ER2V2] Warmup failed: %s", e)


def er2v2_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    audio_chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> np.ndarray:
    num_chunks = len(audio_chunks)
    emb_dim = int(config.embedding_dim)

    if num_chunks == 0:
        return np.zeros((0, emb_dim), dtype=np.float32)

    # Frontend / CMVN 設定をモデルから復元
    spec, _ = load_spec_for_model(model_path)
    cmvn_policy = resolve_cmvn_policy(model_path.parent)
    featurizer = Featurizer(spec)

    # ER2V2 は短すぎる入力で embedding が崩れるので minimum frame を設ける
    min_frames = getattr(config, "min_frames", 40)

    LOGGER.info(
        "[ER2V2] Starting embedding extraction: chunks=%d, sr=%d, min_frames=%d",
        num_chunks, sr, min_frames,
    )

    # まず全チャンクを fbank 特徴に変換
    features = []
    valid_indices = []
    for idx, chunk in enumerate(audio_chunks):
        if chunk is None or len(chunk) == 0:
            features.append(None)
            continue

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug("[ER2V2] Skip chunk=%d: feat is None or too short (frames=%s)",
                         idx, None if feat is None else feat.shape[0])
            features.append(None)
            continue

        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    if not valid_indices:
        LOGGER.warning("[ER2V2] No valid feature chunks. Returning all-zero embeddings.")
        return final_embeddings

    valid_features = [features[i] for i in valid_indices]
    feat_lens = [f.shape[0] for f in valid_features]

    # === NEW: mel_dim を堅牢に決定 ===
    mel_dim = int(valid_features[0].shape[1])
    for k in ("num_mels", "n_mels", "mel_bins", "feature_dim", "dim"):
        if hasattr(spec, k):
            try:
                v = int(getattr(spec, k))
                if v > 0:
                    mel_dim = v
                    break
            except Exception:
                pass
    if mel_dim <= 0:
        mel_dim = 80
    LOGGER.info("[ER2V2] batching with mel_dim=%d", mel_dim)

    order = np.argsort(feat_lens)
    buffer = np.zeros((len(valid_features), emb_dim), dtype=np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    batch_cap = int(getattr(config, "batch_cap", 16))
    bucket_step = int(getattr(config, "bucket_step", 40))

    batch_start = 0
    while batch_start < len(order):
        curr_len = feat_lens[order[batch_start]]
        bucket_max = ((curr_len // bucket_step) + 1) * bucket_step

        batch_end = batch_start
        while (
            batch_end < len(order)
            and feat_lens[order[batch_end]] <= bucket_max
            and (batch_end - batch_start) < batch_cap
        ):
            batch_end += 1

        idxs = order[batch_start:batch_end]
        max_len = max(feat_lens[i] for i in idxs)
        batch_size = len(idxs)

        # === CHANGED: spec.num_mels → mel_dim
        batch_input = np.zeros((batch_size, max_len, mel_dim), dtype=np.float32)

        # === CHANGED: 列数不一致の安全コピー + 最終フレーム繰り返しで時間方向をパディング
        for bi, fi in enumerate(idxs):
            feat = valid_features[fi]  # (T_i, F_feat)
            t, f = feat.shape
            use_cols = min(mel_dim, f)
            batch_input[bi, :t, :use_cols] = feat[:, :use_cols]
            if t < max_len:
                last_frame = batch_input[bi, t - 1 : t, :use_cols]
                repeat = max_len - t
                batch_input[bi, t:, :use_cols] = last_frame.repeat(repeat, axis=0)

        LOGGER.debug("[ER2V2] Inference batch: size=%d, max_len=%d, bucket_max=%d",
                     batch_size, max_len, bucket_max)

        try:
            outputs = session.run([output_name], {input_name: batch_input})[0]
        except Exception as e:
            LOGGER.error("[ER2V2] Inference failed on batch [%d:%d]: %s",
                         batch_start, batch_end, e)
            batch_start = batch_end
            continue

        if outputs.ndim == 3:
            outputs = outputs.mean(axis=1)

        if outputs.shape[0] != batch_size or outputs.shape[1] != emb_dim:
            LOGGER.warning("[ER2V2] Unexpected output shape: got=%s, expect=(%d, %d)",
                           outputs.shape, batch_size, emb_dim)

        for bi, fi in enumerate(idxs):
            if bi < outputs.shape[0]:
                buffer[fi, :] = outputs[bi]

        batch_start = batch_end

    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    LOGGER.info("[ER2V2] Embedding extraction done: valid=%d / %d, emb_dim=%d",
                len(valid_indices), num_chunks, emb_dim)
    return final_embeddings
