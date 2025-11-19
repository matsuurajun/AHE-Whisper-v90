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


# === Wespeaker ResNet293-LM (VoxCeleb) 用ラッパー =======================
from typing import List, Tuple
import numpy as np
import onnxruntime as ort
from pathlib import Path

from ahe_whisper.features import Featurizer
from ahe_whisper.frontend_spec import load_spec_for_model, resolve_cmvn_policy
from ahe_whisper.config import EmbeddingConfig
from ahe_whisper.utils import safe_l2_normalize

# LOGGER はファイル先頭で定義済みのものをそのまま使う想定:
# LOGGER = logging.getLogger("ahe_whisper_worker")


def build_resnet293_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    Wespeaker / wespeaker-voxceleb-resnet293-LM の ONNX セッションを構築。
    CoreMLExecutionProvider が使えればそれを優先。
    """
    sess_options = ort.SessionOptions()
    if getattr(config, "intra_threads", None) is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if getattr(config, "inter_threads", None) is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if getattr(config, "prefer_coreml_ep", False) and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[ResNet293] Building InferenceSession: model=%s, providers=%s, intra=%s, inter=%s",
        str(model_path),
        providers,
        getattr(config, "intra_threads", None),
        getattr(config, "inter_threads", None),
    )
    sess = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    LOGGER.info(
        "[ResNet293] Session built. Inputs=%s, Outputs=%s",
        sess.get_inputs(),
        sess.get_outputs(),
    )
    return sess


def _detect_resnet293_input_layout(session: ort.InferenceSession, feat_dim: int) -> str:
    """
    Wespeaker ResNet293 ONNX の入力テンソルレイアウトを推定する。

    典型パターン:
      - (B, T, F)  例: (None, None, 80)
      - (B, F, T)  例: (None, 80, None)
      - (B, 1, F, T) / (B, 1, T, F)
    """
    in0 = session.get_inputs()[0]
    shape = list(in0.shape)
    ndim = len(shape)

    if ndim == 3:
        # (B, T, F) or (B, F, T)
        _, d1, d2 = shape
        if d2 in (-1, feat_dim):
            return "BTF"  # (B, T, F)
        if d1 in (-1, feat_dim):
            return "BFT"  # (B, F, T)
        # よく分からない場合は最後の軸を特徴量とみなす
        return "BTF"

    if ndim == 4:
        # (B, 1, F, T) or (B, 1, T, F)
        _, _, d1, d2 = shape
        if d1 in (-1, feat_dim):
            return "B1FT"
        if d2 in (-1, feat_dim):
            return "B1TF"
        return "B1FT"

    raise RuntimeError(f"[ResNet293] Unsupported input rank: {shape}")


def _run_resnet293_single(
    session: ort.InferenceSession,
    layout: str,
    input_name: str,
    feat: np.ndarray,
) -> np.ndarray:
    """
    単一チャンク分の特徴量 (T, F) から 1 本の埋め込みベクトルを得る。
    """
    x = feat.astype(np.float32, copy=False)

    if layout == "BTF":
        x = x[None, :, :]          # (1, T, F)
    elif layout == "BFT":
        x = x.T[None, :, :]        # (1, F, T)
    elif layout == "B1FT":
        x = x.T[None, None, :, :]  # (1, 1, F, T)
    elif layout == "B1TF":
        x = x[None, None, :, :]    # (1, 1, T, F)
    else:
        raise RuntimeError(f"[ResNet293] Unknown layout: {layout}")

    y = session.run(None, {input_name: x})[0]
    emb = np.squeeze(y, axis=0)  # (D,) or (1, D) → (D,)
    return emb.reshape(-1)


def resnet293_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wespeaker ResNet293-LM (VoxCeleb) 向けの埋め込み抽出ラッパー。

    Parameters
    ----------
    session:
        build_resnet293_session で作った onnxruntime.InferenceSession
    model_path:
        ONNX ファイルのパス（ログ用）
    chunks:
        1 チャンクごとの波形 (np.ndarray, mono, float32, [-1, 1])
    sr:
        サンプリングレート (想定: 16k)
    config:
        EmbeddingConfig。embedding_dim/min_frames/spec_name/cmvn_policy などを利用。

    Returns
    -------
    embeddings: np.ndarray
        形状 (N, D)。invalid なチャンクの行はゼロ。
    valid_mask: np.ndarray[bool]
        有効なチャンクだけ True。
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        emb_dim0 = getattr(config, "embedding_dim", 192)
        return np.zeros((0, emb_dim0), np.float32), np.zeros((0,), bool)

    # --- frontend 設定 ---
    # model_path から frontend_spec を決める（AHE の既存ロジックに合わせる）
    spec_info = load_spec_for_model(model_path)

    # load_spec_for_model が (spec, default_cmvn) のタプルを返す場合に対応
    if isinstance(spec_info, tuple):
        spec, default_cmvn = spec_info
    else:
        spec = spec_info
        default_cmvn = None

    cmvn_policy = resolve_cmvn_policy(
        spec,
        getattr(config, "cmvn_policy", default_cmvn),
    )
    featurizer = Featurizer(spec)


    min_frames = int(max(getattr(config, "min_frames", 1), 1))

    # --- wave → 特徴量 (T, F) ---
    features: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, chunk in enumerate(chunks):
        if chunk is None or len(chunk) == 0:
            LOGGER.debug("[ResNet293] Skip chunk=%d: empty", idx)
            features.append(None)
            continue

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug(
                "[ResNet293] Skip chunk=%d: feat is None or too short (frames=%s)",
                idx,
                None if feat is None else feat.shape[0],
            )
            features.append(None)
            continue

        # Wespeaker 側の想定に合わせて (T, F) で保持
        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    if not valid_indices:
        LOGGER.warning("[ResNet293] No valid chunks. num_chunks=%d", num_chunks)
        emb_dim0 = getattr(config, "embedding_dim", 192)
        return np.zeros((num_chunks, emb_dim0), np.float32), np.zeros(num_chunks, bool)

    # --- 入力レイアウトと emb_dim を決定 ---
    first_feat = next(f for f in features if f is not None)
    feat_dim = int(first_feat.shape[1])
    layout = _detect_resnet293_input_layout(session, feat_dim)
    in_name = session.get_inputs()[0].name

    # --- 出力埋め込み次元をセッションから決定 ---
    out0 = session.get_outputs()[0]
    emb_dim = 0
    if out0.shape and out0.shape[-1] not in (-1, None):
        emb_dim = int(out0.shape[-1])

    if not emb_dim:
        LOGGER.info("[ECAPA512] Probing emb_dim via dummy forward...")
        emb_probe = _run_resnet293_single(session, layout, in_name, first_feat)
        emb_dim = int(emb_probe.shape[-1])

    # config 側にも反映（ログや他の場所で使うかもしれないので）
    setattr(config, "embedding_dim", emb_dim)

    buffer = np.zeros((len(valid_indices), emb_dim), dtype=np.float32)

    # --- まずは 1 チャンクずつ推論 ---
    for bi, orig_idx in enumerate(valid_indices):
        feat = features[orig_idx]
        if feat is None:
            continue
        emb = _run_resnet293_single(session, layout, in_name, feat)

        if emb.shape[-1] != emb_dim:
            LOGGER.error(
                "[ECAPA512] emb_dim mismatch for chunk=%d: got=%d, expected=%d (this should not happen)",
                orig_idx,
                emb.shape[-1],
                emb_dim,
            )
            # 長さが合わないとバッファに入れられないので、安全のためスキップ
            continue

        buffer[bi, :] = emb

    # --- 元の順序に戻す＆正規化 ---
    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True

    LOGGER.info(
        "[ResNet293] Embedding extraction done: valid=%d / %d, emb_dim=%d",
        len(valid_indices),
        num_chunks,
        emb_dim,
    )
    return final_embeddings, valid_mask


# === Wespeaker ECAPA-TDNN512-LM 用ラッパー =======================
# モデル名の例: wespeaker-ecapa-tdnn512-LM / voxceleb_ECAPA512_LM.onnx

def build_ecapa512_session(model_path: Path, config: EmbeddingConfig) -> ort.InferenceSession:
    """
    Wespeaker ECAPA-TDNN512-LM の ONNX セッションを構築。
    中身は ResNet293 版と同じで、ログのラベルだけ変えています。
    """
    sess_options = ort.SessionOptions()
    if getattr(config, "intra_threads", None) is not None:
        sess_options.intra_op_num_threads = int(config.intra_threads)
    if getattr(config, "inter_threads", None) is not None:
        sess_options.inter_op_num_threads = int(config.inter_threads)

    providers = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if getattr(config, "prefer_coreml_ep", False) and "CoreMLExecutionProvider" in available:
        providers.insert(0, "CoreMLExecutionProvider")

    LOGGER.info(
        "[ECAPA512] Building InferenceSession: model=%s, providers=%s, intra=%s, inter=%s",
        str(model_path),
        providers,
        getattr(config, "intra_threads", None),
        getattr(config, "inter_threads", None),
    )
    sess = ort.InferenceSession(str(model_path), sess_options, providers=providers)
    LOGGER.info(
        "[ECAPA512] Session built. Inputs=%s, Outputs=%s",
        sess.get_inputs(),
        sess.get_outputs(),
    )
    return sess


def ecapa512_embed_batched(
    session: ort.InferenceSession,
    model_path: Path,
    chunks: List[np.ndarray],
    sr: int,
    config: EmbeddingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wespeaker ECAPA-TDNN512-LM 向けの埋め込み抽出ラッパー。

    Parameters
    ----------
    session:
        build_ecapa512_session で作った onnxruntime.InferenceSession
    model_path:
        ONNX ファイルのパス（frontend_spec 判定にも使う）
    chunks:
        1 チャンクごとの波形 (np.ndarray, mono, float32, [-1, 1])
    sr:
        サンプリングレート (想定: 16k)
    config:
        EmbeddingConfig。embedding_dim/min_frames などを利用。

    Returns
    -------
    embeddings: np.ndarray
        形状 (N, D)。invalid なチャンクの行はゼロ。
    valid_mask: np.ndarray[bool]
        有効なチャンクだけ True。
    """
    num_chunks = len(chunks)
    if num_chunks == 0:
        emb_dim0 = getattr(config, "embedding_dim", 512)
        return np.zeros((0, emb_dim0), np.float32), np.zeros((0,), bool)

    # --- frontend 設定 (model_path ベースで spec を決める) ---
    spec_info = load_spec_for_model(model_path)
    if isinstance(spec_info, tuple):
        spec, default_cmvn = spec_info
    else:
        spec = spec_info
        default_cmvn = None

    cmvn_policy = resolve_cmvn_policy(
        spec,
        getattr(config, "cmvn_policy", default_cmvn),
    )
    featurizer = Featurizer(spec)

    min_frames = int(max(getattr(config, "min_frames", 1), 1))

    # --- wave → 特徴量 (T, F) ---
    features: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, chunk in enumerate(chunks):
        if chunk is None or len(chunk) == 0:
            LOGGER.debug("[ECAPA512] Skip chunk=%d: empty", idx)
            features.append(None)
            continue

        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug(
                "[ECAPA512] Skip chunk=%d: feat is None or too short (frames=%s)",
                idx,
                None if feat is None else feat.shape[0],
            )
            features.append(None)
            continue

        feat = feat.astype(np.float32, copy=False)  # (T, F)
        features.append(feat)
        valid_indices.append(idx)

    if not valid_indices:
        LOGGER.warning("[ECAPA512] No valid chunks. num_chunks=%d", num_chunks)
        emb_dim0 = getattr(config, "embedding_dim", 512)
        return np.zeros((num_chunks, emb_dim0), np.float32), np.zeros(num_chunks, bool)

    # --- 入力レイアウトと emb_dim を決定 ---
    first_feat = next(f for f in features if f is not None)
    feat_dim = int(first_feat.shape[1])

    # ResNet293 用の汎用ヘルパーをそのまま流用
    layout = _detect_resnet293_input_layout(session, feat_dim)
    in_name = session.get_inputs()[0].name

    emb_dim = getattr(config, "embedding_dim", 0)
    if not emb_dim:
        out0 = session.get_outputs()[0]
        if out0.shape and out0.shape[-1] not in (-1, None):
            emb_dim = int(out0.shape[-1])
        else:
            LOGGER.info("[ECAPA512] Probing emb_dim via dummy forward...")
            emb_probe = _run_resnet293_single(session, layout, in_name, first_feat)
            emb_dim = int(emb_probe.shape[-1])

    buffer = np.zeros((len(valid_indices), emb_dim), dtype=np.float32)

    # --- まずは 1 チャンクずつ推論 ---
    for bi, orig_idx in enumerate(valid_indices):
        feat = features[orig_idx]
        if feat is None:
            continue
        emb = _run_resnet293_single(session, layout, in_name, feat)
        if emb.shape[-1] != emb_dim:
            LOGGER.warning(
                "[ECAPA512] emb_dim mismatch for chunk=%d: got=%d, expected=%d",
                orig_idx,
                emb.shape[-1],
                emb_dim,
            )
            emb = emb[:emb_dim]
        buffer[bi, :] = emb

    # --- 元の順序に戻す＆正規化 ---
    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    for i, orig_idx in enumerate(valid_indices):
        final_embeddings[orig_idx, :] = buffer[i]

    final_embeddings = safe_l2_normalize(final_embeddings.astype(np.float32))
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True

    LOGGER.info(
        "[ECAPA512] Embedding extraction done: valid=%d / %d, emb_dim=%d",
        len(valid_indices),
        num_chunks,
        emb_dim,
    )
    return final_embeddings, valid_mask

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
        return (
            np.zeros((0, emb_dim), dtype=np.float32),
            np.zeros(0, dtype=bool),
        )

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
        
        # 【追加修正 1】 音声のスケールを Kaldi/WeSpeaker 仕様 (16bit scale) に合わせる
        # librosa は [-1, 1] ですが、モデルが [-32768, 32767] を期待している可能性が大です
        chunk = chunk * 32768.0

        # ここで特徴量を取得
        feat = featurizer.get_mel_spectrogram(chunk, sr, cmvn_policy)
        
        if feat is None or feat.shape[0] < min_frames:
            LOGGER.debug("[ER2V2] Skip chunk=%d: too short", idx)
            features.append(None)
            continue
        
        # 【重要修正】 強制的にインスタンス正規化 (CMVN) を適用する
        # 平均を引いて、標準偏差で割ります。1e-6 はゼロ除算防止です。
        # axis=0 は「時間方向」の平均・分散を計算します。
        mean = feat.mean(axis=0)
        std = feat.std(axis=0)
        feat = (feat - mean) / (std + 1e-6)

        feat = feat.astype(np.float32, copy=False)
        features.append(feat)
        valid_indices.append(idx)

    final_embeddings = np.zeros((num_chunks, emb_dim), dtype=np.float32)
    if not valid_indices:
        LOGGER.warning("[ER2V2] No valid feature chunks. Returning all-zero embeddings.")
        valid_mask = np.zeros(num_chunks, dtype=bool)
        return final_embeddings, valid_mask

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
    valid_mask = np.zeros(num_chunks, dtype=bool)
    valid_mask[valid_indices] = True
    
    return final_embeddings, valid_mask
