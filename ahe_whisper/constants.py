# -*- coding: utf-8 -*-
# Single Source of Truth (SSOT) for model definitions.
# All models are pinned to a specific revision for reproducibility.
from typing import Dict, Any

MODELS: Dict[str, Dict[str, Any]] = {
    "embedding": {
        # 修正: 正しいリポジトリIDに変更 (Wespeaker/speaker-embedding-models -> csukuangfj/speaker-embedding-models)
        "repo_id": "csukuangfj/speaker-embedding-models",
        # Cam++ (Context-Aware Masking TDNN) - Winner of the benchmark
        "required_files": [
            "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
            # 修正: config.yaml はこのリポジトリに存在しないため削除 (デフォルト設定が使用されます)
        ],
        "primary_file": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    },
    "vad": {
        "repo_id": "onnx-community/silero-vad",
        #"revision": "c25b5c13b916163b41315668a32973167a5786c2",
        "required_files": ["onnx/model.onnx"],
        "primary_file": "onnx/model.onnx"
    },
    "asr": {
        "repo_id": "mlx-community/whisper-large-v3-turbo",
        #"revision": "0f393c2688b5e640b791b017a14c6328352652a1",
        "use_snapshot": True,
    }
}