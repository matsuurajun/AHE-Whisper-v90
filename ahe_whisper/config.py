# -*- coding: utf-8 -*-
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class TranscriptionConfig:
    model_name: str = "mlx-community/whisper-large-v3-turbo"
    language: Optional[str] = "ja"
    no_speech_threshold: float = 0.4
    logprob_threshold: float = -1.0

@dataclass
class EmbeddingConfig:
    embedding_dim: int = 192
    batch_cap: int = 4
    bucket_step: int = 80
    prefer_coreml_ep: bool = False
    intra_threads: Optional[int] = None
    inter_threads: int = 1
    embedding_win_sec: float = 1.5
    embedding_hop_sec: float = 0.75
    # VAD の平均スコアがこの値以上のチャンクだけを
    # 「話者クラスタリングに使う embedding」として残す
    min_chunk_speech_prob: float = 0.30
    # 時間方向スムージング用カーネル
    # 0 → スムージング無効
    # 3,5 など奇数 → 前後チャンクを含む移動平均
    smooth_embeddings_kernel: int = 0

@dataclass
class DiarizationConfig:
    min_speakers: int = 2
    max_speakers: int = 4
    engine: str = "soft-em-adc"
    vad_th_start: float = 0.50
    vad_th_end: float = 0.20
    em_tau_schedule: List[float] = field(default_factory=lambda: [10.0, 5.0, 3.0])
    min_speaker_duration_sec: float = 1.2
    min_fallback_duration_sec: float = 1.0
    min_speech_sec: float = 0.3
    max_merge_gap_sec: float = 1.5
    # --- NEW: cluster post-processing knobs ---
    # クラスタが担当する embedding 比率がこれ未満なら「小さすぎる」とみなして候補から外す
    min_cluster_mass: float = 0.05
    # セントロイド同士のコサイン類似度がこれ以上なら「同一話者」とみなしてマージ
    centroid_merge_sim: float = 0.90

    def __post_init__(self) -> None:
        if not (0.0 <= self.vad_th_start <= 1.0):
            raise ValueError(f"vad_th_start must be in [0, 1], got {self.vad_th_start}")
        if not (0.0 <= self.vad_th_end <= 1.0):
            raise ValueError(f"vad_th_end must be in [0, 1], got {self.vad_th_end}")
        if self.vad_th_start < self.vad_th_end:
            raise ValueError(f"vad_th_start ({self.vad_th_start}) must be >= vad_th_end ({self.vad_th_end}).")
        if self.min_speakers > self.max_speakers:
            raise ValueError(f"min_speakers ({self.min_speakers}) cannot be greater than max_speakers ({self.max_speakers}).")
        if not (0.0 < self.min_cluster_mass <= 1.0):
            raise ValueError(f"min_cluster_mass must be in (0, 1], got {self.min_cluster_mass}")
        if not (0.0 < self.centroid_merge_sim <= 1.0):
            raise ValueError(f"centroid_merge_sim must be in (0, 1], got {self.centroid_merge_sim}")

@dataclass
class VadConfig:
    window_size_samples: int = 320

@dataclass
class AlignerConfig:
    # VAD / spk_probs / word_cost の重み
    alpha: float = 0.6    # VAD 重み（そのまま維持）
    beta: float = 0.3    # 話者確率 spk_probs の重み
    gamma: float = 1.0    # word_cost の重み

    # 話者スイッチのペナルティ
    delta_switch: float = 0.08

    non_speech_th: float = 0.02
    grid_hz: int = 50

    # --- v90.97 Smooth Aligner options ---
    # diarizer 側ですでに平滑＋sharp を掛けているので、
    # ひとまず OverlapDPAligner 側ではオフにして挙動を素直に見る。
    use_smooth_aligner: bool = False
    smooth_alpha: float = 0.55
    smooth_gamma: float = 1.4

    def __post_init__(self) -> None:
        if not (0.0 <= self.non_speech_th <= 1.0):
            raise ValueError(f"non_speech_th must be in [0, 1], got {self.non_speech_th}")
        if self.grid_hz < 1:
            raise ValueError(f"grid_hz must be >= 1, got {self.grid_hz}")

@dataclass
class ExportConfig:
    output_formats: List[str] = field(default_factory=lambda: ["json", "srt", "txt"])
    srt_max_line_width: int = 38

@dataclass
class AppConfig:
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    aligner: AlignerConfig = field(default_factory=AlignerConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    output_dir: str = "AHE-Whisper-output"

    def to_dict(self) -> dict:
        return asdict(self)
