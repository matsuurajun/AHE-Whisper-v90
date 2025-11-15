# ahe_whisper/phrase_grouping.py
from typing import List, Dict, Any

def soft_phrase_grouping(
    words: List[Dict[str, Any]],
    gap_th: float = 0.45,
    max_len: int = 12,
) -> List[Dict[str, Any]]:
    """
    Whisper の word-level を phrase-level に “軽く”まとめる。
    DP-Aligner の誤切替を抑制するための前処理。

    words: [{"word": str, "start": float, "end": float}, ...]
    """
    if not words:
        return words

    phrases = []
    cur = [words[0]]

    for w in words[1:]:
        prev = cur[-1]

        # 1) 時間ギャップが短い → 同じフレーズにまとめる
        if (w["start"] - prev["end"]) < gap_th:
            cur.append(w)
            continue

        # 2) フレーズが長すぎる場合は切る
        if len(cur) >= max_len:
            phrases.append(cur)
            cur = [w]
            continue

        # 3) 通常のフレーズ切り替え
        phrases.append(cur)
        cur = [w]

    phrases.append(cur)

    # 出力を phrase 用 words に flatten
    # （Aligner には word/phrase 単位の time-span が渡れば良い）
    grouped_words: List[Dict[str, Any]] = []
    for ph in phrases:
        if not ph:
            continue
        grouped_words.append({
            "word": "".join([w["word"] for w in ph]),
            "start": ph[0]["start"],
            "end": ph[-1]["end"],
            "children": ph,  # 元の word list を保持（必要に応じて使える）
        })

    return grouped_words


def phrase_group_words(
    words: List[Dict[str, Any]],
    mode: str = "balanced",
) -> List[Dict[str, Any]]:
    """
    pipeline から呼ばれる公開 API。
    現状は mode="balanced" を標準とし、
    将来 tight/loose 等を追加できるようにしておく。
    """
    if not words:
        return words

    if mode == "balanced":
        # デフォルト設定（今の soft_phrase_grouping と同じ）
        return soft_phrase_grouping(words, gap_th=0.45, max_len=12)
    elif mode == "tight":
        # もう少し細かく切るバリエーション（将来用）
        return soft_phrase_grouping(words, gap_th=0.30, max_len=8)
    elif mode == "loose":
        # もう少し長くまとめるバリエーション（将来用）
        return soft_phrase_grouping(words, gap_th=0.60, max_len=16)
    else:
        # 未知の mode が来ても safe fallback
        return soft_phrase_grouping(words, gap_th=0.45, max_len=12)
