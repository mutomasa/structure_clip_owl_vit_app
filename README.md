Structure-CLIP + OWL‑ViT Streamlit App
=====================================

このリポジトリは、Structure-CLIP の CLIP 互換実装によるテキスト×画像の類似度推論と、OWL‑ViT によるテキスト条件ゼロショット検出（バウンディングボックス描画）、およびシーングラフ（主語-関係-目的語）の可視化を行う Streamlit アプリです。

参考: [zjukg/Structure-CLIP レポジトリ](https://github.com/zjukg/Structure-CLIP)

セットアップ
------------

1. 依存関係のインストール（`uv` 利用）

```bash
uv venv
uv pip install -e .
```

2. Structure-CLIP のコードをサブモジュールとして取得

```bash
git clone https://github.com/zjukg/Structure-CLIP third_party/Structure-CLIP
```

3. 起動

```bash
uv run streamlit run app.py
```

機能
----

- 画像アップロードとテキスト入力
- Structure-CLIP 同梱の CLIP 実装（`third_party/Structure-CLIP/model/clip.py`）による類似度推論
- OWL‑ViT によるテキスト条件ゼロショット検出とバウンディングボックス描画（Hugging Face pipeline）
- シーングラフ（主語-関係-目的語）のGraphviz可視化

注意
----

- 元論文・実装では追加のデータ前処理や学習済みチェックポイントが必要です。ここでは最小実行デモとして、可能な限り軽量化して利用します。


技術的な詳細
--------------

### 全体構成

- UI: `Streamlit` ベース（`app.py`）
- 類似度推論: Structure-CLIP レポジトリ同梱の `clip.py`（OpenAI CLIP 互換実装）を直接 import して利用
- 検出: Hugging Face `pipeline("zero-shot-object-detection")`（`google/owlvit-base-patch32`）を利用
- 可視化: `st.image` によるボックス描画画像の表示、`st.graphviz_chart` によるシーングラフ描画

### Structure-CLIP について（技術概説）

- 論文: Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations（AAAI 2024）
- 目的: 画像・テキストの表現にシーングラフ知識（主語-関係-目的語）を統合し、構造化表現を強化
- 主要構成（レポジトリより）:
  - `model/model.py`: CLIP 互換の視覚バックボーン（ResNet/ViT）とテキスト Transformer に加え、関係表現のための Transformer を備えた実装
  - `model/clip.py`: OpenAI CLIP チェックポイントのロード・前処理・トークナイズ（`tokenize`）を含む互換モジュール
- 本アプリでの利用方針:
  - まずは軽量な「テキスト-画像 類似度」デモにフォーカスし、`clip.load("ViT-B/32")` の推論を使用
  - フルのシーングラフ強化（学習済み checkpoint を用いた構造統合）は任意拡張として今後対応可能

参考: [zjukg/Structure-CLIP (GitHub)](https://github.com/zjukg/Structure-CLIP)

### OWL‑ViT について（技術概説）

- モデル: OWL‑ViT（Open‑World Localization with Vision Transformers）
- 特徴: ラベル語彙に依存せず、テキスト候補を与えるだけでゼロショット検出を実行可能（ローカライズ能力）
- 本アプリでの利用:
  - `transformers` の `pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")` を使用
  - 入力画像とテキスト候補（カンマ区切り）を与え、`[{label, score, box}]` を取得
  - `PIL.ImageDraw` でバウンディングボックス＋ラベルを描画し、`st.image` で表示

### シーングラフ可視化

- 入力フォーマット（1行1トリプル）:
  - `subject, relation, object`
  - `subject ->relation-> object`
- アプリ内でパースし、DOT 形式に変換して `st.graphviz_chart` で描画
- 例: `dog, on, couch`／`person ->holding-> phone`

### 推論フロー（簡略）

1. 画像アップロード、テキスト候補入力、（任意で）構造トリプル入力
2. CLIP によるテキスト-画像類似度計算（ランキング表示）
3. OWL‑ViT による検出とボックス描画（有効化時）
4. 構造トリプルのGraphviz可視化

### 依存関係（主なもの）

- `streamlit`, `torch`, `torchvision`, `transformers`, `accelerate`, `numpy`, `Pillow`, `timm`, `ftfy`, `regex`
- `uv` により `pyproject.toml` の依存をインストール

### 制約・注意点

- OWL‑ViT/CLIP のモデルダウンロード時に初回のみネットワークアクセスが発生
- CPU 実行は可能だが、検出は GPU 環境推奨（遅延・メモリに注意）
- Structure‑CLIP 本来のシーングラフ強化学習を再現するには追加データと checkpoint が必要

### 実行例

```bash
uv run streamlit run app.py --server.headless true --server.port 8501
```



