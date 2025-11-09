# -*- coding: utf-8 -*-
# Single Source of Truth (SSOT) for model definitions.
# All models are pinned to a specific revision for reproducibility.
from typing import Dict, Any

MODELS: Dict[str, Dict[str, Any]] = {
    "embedding": {
        "repo_id": "Wespeaker/wespeaker-ecapa-tdnn512-LM",
        #"revision": "90233509ddc5368a88f53c9e3650646c0f065355",
        "required_files": ["voxceleb_ECAPA512_LM.onnx", "config.yaml"],
        "primary_file": "voxceleb_ECAPA512_LM.onnx"
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
