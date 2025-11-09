# AHE-Whisper

**Adaptive Hybrid Engine for Whisper — v90.90（Definitive Release）**
作成者：松浦 純（Matsuura Jun）
日付：2025年11月

---

## 🧭 概要

**AHE-Whisper** は、Apple Silicon向けに最適化されたエンドツーエンドの音声認識・話者分離エンジンです。
複数のASR（音声認識）、VAD（音声区間検出）、話者埋め込みモデルをモジュール化し、組み合わせ可能な形で統合。
TV番組制作（例：*NHK WORLD「BIZ STREAM」*）などの実運用を想定し、**速度・精度・再現性を最大化**する設計となっています。

---

## ⚙️ コア構成

| モジュール         | 役割        | 使用モデル                                 |
| ------------- | --------- | ------------------------------------- |
| **ASR**       | 音声認識      | Whisper Large-v3-Turbo（MLX最適化版）       |
| **VAD**       | 音声区間検出    | Silero-VAD（ONNX）                      |
| **Embedding** | 話者埋め込み    | WeSpeaker ECAPA-TDNN-512              |
| **Aligner**   | セグメント整列   | OverlapDPAligner（独自）                  |
| **Clusterer** | 話者クラスタリング | Attractor-based Deep Clustering（試験実装） |
| **Frontend**  | UI層       | niceGUI v9.x（Python 3.12 / MLXスタック）   |

---

## 🧩 ディレクトリ構成

```
AHE-Whisper/
├── ahe_whisper/               # コアエンジンモジュール
├── tools/                     # ベンチマーク・診断ツール
├── models/                    # （除外対象）ローカルモデルキャッシュ
├── setup_ahe_whisper.py       # プロジェクトセットアップスクリプト
├── requirements.txt           # 依存ライブラリ
├── run_offline.command        # macOS用起動スクリプト
├── README.md
└── CHANGELOG.md               # バージョン履歴
```

> ⚠️ `models/` フォルダは `.gitignore` により除外されています。
> Whisper や VAD モデルは `prefetch_models.py` または `setup_ahe_whisper.py` で自動取得できます。

---

## 🚀 セットアップ（Apple Silicon）

```bash
cd AHE-Whisper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python setup_ahe_whisper.py --init
```

または **uv** を使用する場合：

```bash
uv venv
uv pip install -r requirements.txt
uv run setup_ahe_whisper.py --init
```

---

## 🧠 主な特徴

* 🔹 MLXによるWhisper推論の高速化（Mシリーズ最適化）
* 🔹 自動VAD×ASR統合のハイブリッドエンジン
* 🔹 OverlapDPAligner＋Attractorクラスタリングによる重複話者対応
* 🔹 話者区別付き書き起こし（`.srt`, `.vtt`, `.json`対応）
* 🔹 `setup_ahe_whisper.py` による再現性確保
* 🔹 完全オフライン動作（外部API不要）

---

## 🧪 使用例

```bash
python setup_ahe_whisper.py --transcribe input_audio.mp3
```

出力例：

```
AHE-Whisper-output/
 ├── transcript.json
 ├── transcript.vtt
 └── speaker_timeline.csv
```

---

## 📄 ライセンス

モデルファイルはそれぞれの配布元ライセンス（OpenAI / ONNX-Community / WeSpeaker）に従います。
カスタムコード © 2025 Matsuura Jun.
本リポジトリは研究・開発目的であり、モデル再配布は行いません。

---

## 🏷️ バージョン情報

現在のリリース：**v90.90 – Definitive**
タグ：`v90.90`
日付：2025-11-08
次期マイルストーン：**v91.00（Fluid Inference Beta）**

---

# 🪶 更新履歴（CHANGELOG）

## [v90.90] — 2025-11-08

**Definitive Release（最終確定版）**

* すべてのモジュールを統合した `setup_ahe_whisper.py` を実装
* OverlapDPAligner によるハイブリッドVAD-ASR推論を導入
* WeSpeaker ECAPA-TDNN512（ONNX）を話者埋め込みに統合
* MLX Whisper 推論を最適化（M4 24GB環境でRTF ≈ 0.27）
* セッションキャッシュの再現性向上
* niceGUI v9.5 によるGUI最終版
* `.gitignore` にモデル出力除外設定を追加
* MシリーズMacでの再現性を確認

---

## [v83.12] — 2025-09-14

* Silero-VADに適応型エネルギー正規化を追加
* マルチスピーカー音声で話者分離精度を改善
* 内部ログ機能 `TRACE-ALIGNER` モードを導入

---

## [v75.02] — 2025-07-03

* ONNXRuntime による埋め込み推論へ移行
* DP後処理によるバッチセグメント整列を追加

---

## [v71.00] — 2025-05-11

* MLXベースの WhisperKit バックエンドを実装
* オフラインモデル取得スクリプト `prefetch_models.py` を導入

---

## [v34.00] — 2024-11-20

* VAD × ASR ハイブリッドパイプラインの初期統合
* GUI試作版（v3.0）と設定ジェネレーターを追加

---

### 凡例

🧩 **Core**：アーキテクチャやエンジンの主要変更
⚙️ **Perf**：速度・メモリ最適化
🧠 **UX/UI**：操作性・UI改善
📦 **Infra**：環境構築やビルド関連
