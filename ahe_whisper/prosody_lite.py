# ahe_whisper/prosody_lite.py
import numpy as np
import librosa

def extract_prosody_lite(waveform, sr=16000, hop=160):
    """
    ピッチ・エネルギー・休止の3つを軽量推定し、
    phrase grouping の補助情報として返す。
    """
    # pitch（YIN）
    try:
        pitch = librosa.yin(
            waveform.astype(np.float32),
            fmin=50, fmax=550,
            sr=sr, frame_length=1024, hop_length=hop
        )
    except Exception:
        pitch = np.zeros(len(waveform) // hop)

    # energy
    frames = librosa.util.frame(waveform, frame_length=1024, hop_length=hop)
    energy = np.sum(frames**2, axis=0)

    # pause-likelihood（低エネルギーの割合）
    pause = (energy < np.percentile(energy, 20)).astype(float)

    return {
        "pitch": pitch,
        "energy": energy,
        "pause": pause,
        "hop_sec": hop / sr,
    }
