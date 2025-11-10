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

@dataclass
class DiarizationConfig:
    min_speakers: int = 0
    max_speakers: int = 4
    engine: str = "soft-em-adc"
    vad_th_start: float = 0.50
    vad_th_end: float = 0.20
    em_tau_schedule: List[float] = field(default_factory=lambda: [10.0, 5.0, 3.0])
    min_speaker_duration_sec: float = 3.0
    min_fallback_duration_sec: float = 1.0
    min_speech_sec: float = 0.2
    max_merge_gap_sec: float = 1.2

    def __post_init__(self) -> None:
        if not (0.0 <= self.vad_th_start <= 1.0):
            raise ValueError(f"vad_th_start must be in [0, 1], got {self.vad_th_start}")
        if not (0.0 <= self.vad_th_end <= 1.0):
            raise ValueError(f"vad_th_end must be in [0, 1], got {self.vad_th_end}")
        if self.vad_th_start < self.vad_th_end:
            raise ValueError(f"vad_th_start ({self.vad_th_start}) must be >= vad_th_end ({self.vad_th_end}).")
        if self.min_speakers > self.max_speakers:
            raise ValueError(f"min_speakers ({self.min_speakers}) cannot be greater than max_speakers ({self.max_speakers}).")

@dataclass
class VadConfig:
    window_size_samples: int = 320

@dataclass
class AlignerConfig:
    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.5
    delta_switch: float = 0.1
    non_speech_th: float = 0.02
    grid_hz: int = 50

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
