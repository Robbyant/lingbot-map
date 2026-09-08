<div align="center">
  <img src="assets/teaser.webp" width="100%">

<h1>LingBot-Map：ストリーミング 3D 再構成のための幾何コンテキスト Transformer</h1>

Robbyant Team

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

</div>

<div align="center">

[![Paper](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv)](https://arxiv.org/abs/2604.14141)
[![PDF](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=adobeacrobatreader)](lingbot-map_paper.pdf)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://technology.robbyant.com/lingbot-map)
[![HuggingFace](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/robbyant/lingbot-map)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://www.modelscope.cn/models/Robbyant/lingbot-map)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE.txt)

</div>

https://github.com/user-attachments/assets/fe39e095-af2c-4ec9-b68d-a8ba97e505ab

-----

### 🗺️ LingBot-Map を紹介します！ストリーミング 3D 再構成のためのフィードフォワード型 3D 基盤モデルです！🏗️🌍

LingBot-Map は次の点に重点を置いています。

- **幾何コンテキスト Transformer**：アンカーコンテキスト、姿勢参照ウィンドウ、軌跡メモリを用い、座標グラウンディング、密な幾何学的手がかり、長距離ドリフト補正を単一のストリーミングフレームワーク内でアーキテクチャとして統合します。
- **高効率なストリーミング推論**：ページ化 KV キャッシュアテンションを備えたフィードフォワードアーキテクチャにより、518×378 の解像度で 10,000 フレームを超える長いシーケンスに対して約 20 FPS の安定した推論を実現します。
- **最先端の再構成性能**：既存のストリーミング手法および反復最適化ベースの手法と比べ、さまざまなベンチマークで優れた性能を発揮します。

---

## 📑 目次

<details>
<summary>クリックして展開</summary>

- [📰 最新情報](#-最新情報)
- [📋 TODO](#-todo)
- [⚙️ インストール](#️-インストール)
- [📦 モデルのダウンロード](#-モデルのダウンロード)
- [🚀 クイックスタート](#-クイックスタート)
- [🎬 インタラクティブデモ（`demo.py`）](#-インタラクティブデモdemopy)
  - [サンプルシーンを試す](#サンプルシーンを試す)
  - [キーフレーム間隔を用いたストリーミング](#キーフレーム間隔を用いたストリーミング)
  - [ウィンドウ推論（3000 フレームを超える長いシーケンス向け）](#ウィンドウ推論3000-フレームを超える長いシーケンス向け)
  - [空マスク](#空マスク)
  - [可視化オプション](#可視化オプション)
  - [パフォーマンスとメモリ](#パフォーマンスとメモリ)
- [🎥 オフラインレンダリングパイプライン（`demo_render/batch_demo.py`）](#-オフラインレンダリングパイプラインdemo_renderbatch_demopy)
- [📜 ライセンス](#-ライセンス)
- [📖 引用](#-引用)
- [✨ 謝辞](#-謝辞)

</details>

---

## 📰 最新情報

- **2026-06-28** — SDPA KV キャッシュの不具合を修正しました。**SDPA バックエンドは長いシーケンスでより高い性能を発揮するようになりました**。最高の性能を得るには、引き続き FlashInfer バックエンドを推奨します。
- **2026-05-25** — 📊 **評価ベンチマークを公開しました**。KITTI と Oxford Spires の評価スクリプトを公開しました。パイプラインは [benchmark/](benchmark/) を参照し、評価前に [`preprocess/oxford.py`](preprocess/oxford.py) を実行して Oxford Spires のデータを準備してください。
- **2026-04-29** — 📹 **長時間動画のデモを公開しました**。オフラインパイプラインでレンダリングした非常に長い動画の例（約 25,000 フレーム、13 分間の屋内ウォークスルー）を公開しました。コマンド、フラグの理由、レンダリング結果は[実行例](#実行例-長時間の屋内ウォークスルー約-25000-フレーム13-分)を参照してください。
- **2026-04-27** — 🚀 **LingBot-Map を高速化しました**。最新の `main` を取得し、`python demo.py --compile ...` または `python gct_profile.py --backend flashinfer --dtype bf16 --compile` を実行して、お使いのハードウェアで確認してください。
- **2026-04-24** — `--keyframe_interval > 1` のときに非キーフレームが暗黙にキャッシュされる FlashInfer KV キャッシュの不具合を修正しました。**320 フレームを超えて実行する場合の姿勢および再構成品質が改善されます**。

---

## 📋 TODO

- ✅ 評価ベンチマークを公開
  - ✅ Oxford Spires データセット
  - ✅ KITTI データセット
  - ✅ VBR データセット
  - ✅ Droid-W データセット
  - ✅ TUM-D データセット
  - ✅ 7-scenes データセット
  - ✅ ETH3D データセット
  - ✅ Tanks and Temples データセット
  - ✅ NRGBD データセット
- ✅ デモスクリプトを公開
  - ✅ 屋内の長時間動画デモ（[注目の屋内ウォークスルー](#-注目屋内ウォークスルー約-25000-フレーム13-分)）
  - ✅ 屋外の長時間動画デモ
  - ✅ LingBot-World デモ（[実行例](#実行例-lingbot-world-のシーン)）
  - ✅ 空撮の長時間動画デモ

---

## ⚙️ インストール

**1. conda 環境を作成**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. PyTorch（CUDA 12.8）をインストール**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

> PyTorch 2.8.0 を推奨します。バッチレンダリングパイプラインに必要な NVIDIA Kaolin が `torch-2.8.0_cu128` 用のビルド済み wheel を提供しているためです。`demo.py` のみを使用する場合は新しい PyTorch でも構いませんが、その場合、バッチレンダラーでは Kaolin をソースからビルドする必要があります。
> その他の CUDA バージョンについては、[PyTorch Get Started](https://pytorch.org/get-started/locally/) を参照してください。

**3. lingbot-map をインストール**

```bash
pip install -e .
```

**4. FlashInfer をインストール（推奨）**

FlashInfer は、効率的なストリーミング推論のためのページ化 KV キャッシュアテンションを提供します。初回使用時に CUDA カーネルを JIT コンパイルする純粋な Python パッケージなので、1 つの wheel で複数の CUDA/PyTorch バージョンに対応できます。

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

> `--index-url https://pypi.org/simple` が必要なのは、デフォルトの pip インデックスが `flashinfer-python` を含まない内部ミラーである場合のみです。
> （任意）初回使用を高速化するため、CUDA 専用の JIT キャッシュもインストールできます：`pip install flashinfer-jit-cache -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/`。
> 詳細は [FlashInfer のインストール](https://docs.flashinfer.ai/installation.html) を参照してください。FlashInfer がインストールされていない場合、`--use_sdpa` によって SDPA（PyTorch ネイティブアテンション）へフォールバックします。

**5. 可視化用の依存関係（任意）**

```bash
pip install -e ".[vis]"
```

## 📦 モデルのダウンロード

| モデル名 | Hugging Face リポジトリ | ModelScope リポジトリ | 説明 |
| :--- | :--- | :--- | :--- |
| lingbot-map-long | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | 長いシーケンスや大規模なシーンに適しています。 |
| lingbot-map | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | バランス型チェックポイント（論文、ベンチマーク、オフラインデモで使用）。短いシーケンスと長いシーケンスの双方で総合的な性能を両立します。 |
| lingbot-map-stage1 | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | lingbot-map のステージ 1 学習チェックポイント。VGGT モデルへ読み込み、双方向推論（c2w）に使用できます。 |

> 🚧 **近日公開：**より長いシーケンスに対応する、さらに強力なモデルを学習中です。続報をお待ちください。

## 🚀 クイックスタート

インストール後、次の 1 コマンドで最初のシーンを実行します。

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```

これにより、`http://localhost:8080` でインタラクティブな [viser](https://github.com/nerfstudio-project/viser) ビューアーが起動します。シーンとフラグの全一覧は下記の[インタラクティブデモ](#-インタラクティブデモdemopy)を、長いシーケンスのバッチレンダリングは[オフラインレンダリングパイプライン](#-オフラインレンダリングパイプラインdemo_renderbatch_demopy)を参照してください。

## 🎬 インタラクティブデモ（`demo.py`）

`demo.py` を実行すると、ブラウザベースの [viser](https://github.com/nerfstudio-project/viser) ビューアー（デフォルトは `http://localhost:8080`）でインタラクティブな 3D 可視化を行えます。

### サンプルシーンを試す

`example/` には、そのまま実行できる 3 つのサンプルシーンが用意されています。
```bash
# courthouse scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```


https://github.com/user-attachments/assets/aa10f7ab-8024-43c7-92f8-d56159ec85c8






```bash
# University scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/university --mask_sky
```


https://github.com/user-attachments/assets/212a1744-6ff5-4ccf-9bd4-728608248b57







```bash
# Loop scene (loop closure trajectory)
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop
```


https://github.com/user-attachments/assets/5ae0a292-b081-40c6-838c-b7c1a0538d75





#### 🎯 注目：屋内ウォークスルー（約 25,000 フレーム、13 分）


*このシーケンスはインタラクティブな viser ビューアーには長すぎるため、[オフラインレンダリングパイプライン](#-オフラインレンダリングパイプラインdemo_renderbatch_demopy)でレンダリングしました。完全なコマンドは該当セクションを参照してください。*

今後さらに多くのサンプルを提供する予定です。

### 動的シーンのデモ（Droid-W）

**データセット：**Hugging Face の [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) からデモシーケンスをダウンロードします。

上記データセットの `dynamic` シーケンスでの実行例です（空マスクを有効化、カメラ最適化を 4 回反復、2 フレームごとにキーフレーム）。

空マスク、4 回のカメラ最適化反復、入力ストライド 2 で `dynamic` シーケンスを実行します。

```bash
python demo.py \
    --image_folder /path/to/dynamic\
    --model_path ../../Lingbot-Map/lingbot-map.pt \
    --camera_num_iterations 4 \
    --mask_sky \
    --stride 2
```



https://github.com/user-attachments/assets/567b6e9b-1cbf-402a-96be-9bab70715ec3

<img width="1453" height="1195" alt="image" src="https://github.com/user-attachments/assets/27f8c6b7-339e-4e5f-9776-7cb577147401" />





### キーフレーム間隔を用いたストリーミング

`--keyframe_interval` を使用すると、N フレームごとのフレームだけをキーフレームとして保持し、KV キャッシュのメモリ使用量を削減できます。非キーフレームも予測を生成しますが、キャッシュには保存されません。これは 320 フレームを超える長いシーケンスに有効です（320 ビューの video RoPE で学習しているため、KV キャッシュに 320 ビューを超えて保存すると性能が低下します。キーフレーム戦略を使えば、より長いシーケンスを推論できます）。demo.py ではキーフレーム間隔が自動計算されます。

> **推論範囲について。**本手法はデフォルトで状態をリセットしないため、最大推論範囲は学習データセットで観測された最長距離に制限されます。その距離を超える場合は状態のリセットが必要です。姿勢の崩壊が見られたら、ウィンドウモード（`--mode windowed`）へ切り替えてください。多くの場合、`--keyframe_interval` の調整だけで十分であり、その他のウィンドウパラメーターはデフォルト値のまま使用できます。


### ウィンドウ推論（3000 フレームを超える長いシーケンス向け）

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128 --overlap_keyframes 16 --keyframe_interval 2 
```


### 空マスク

空マスクでは ONNX の空セグメンテーションモデルを使用し、再構成した点群から空の点を除外します。これにより、屋外シーンの可視化品質が向上します。

**セットアップ：**

```bash
# Install onnxruntime (required)
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # GPU (faster for large image sets)
```

空セグメンテーションモデル（`skyseg.onnx`）は、初回使用時に [Hugging Face](https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx) から自動的にダウンロードされます。

**使用方法：**

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky
```

空マスクは `<image_folder>_sky_masks/` にキャッシュされるため、2 回目以降の実行では再生成を省略できます。`--sky_mask_dir` で独自のキャッシュディレクトリを指定したり、`--sky_mask_visualization_dir` でマスクの並列比較画像を保存したりすることもできます。

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --sky_mask_dir /path/to/cached_masks/ \
    --sky_mask_visualization_dir /path/to/mask_viz/
```

### 可視化オプション

| 引数 | デフォルト | 説明 |
|:---|:---|:---|
| `--port` | `8080` | Viser ビューアーのポート |
| `--conf_threshold` | `1.5` | 低信頼度の点を除外する可視性しきい値 |
| `--point_size` | `0.00001` | 点群の点サイズ |
| `--downsample_factor` | `10` | 点群表示の空間ダウンサンプリング係数 |

### パフォーマンスとメモリ

#### FlashInfer を使用しない場合（SDPA へフォールバック）

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --use_sdpa
```

#### GPU メモリが限られている環境での実行

メモリ不足が発生した場合は、次のいずれか、または両方を試してください。

- **`--offload_to_cpu`** — 推論中のフレームごとの予測を CPU にオフロードします（デフォルトで有効。メモリに余裕がある場合のみ `--no-offload_to_cpu` を使用してください）。
- **`--num_scale_frames 2`** — 双方向スケールフレーム数をデフォルトの 8 から 2 へ減らし、初期スケールフェーズのアクティベーションピークを抑えます。

#### 推論の高速化

カメラヘッドの反復改良ステップ数を減らすことで、姿勢精度をわずかに犠牲にして実行時間を短縮できます。

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --camera_num_iterations 1
```

`--camera_num_iterations` のデフォルトは `4` です。`1` に設定すると、カメラヘッドで 3 回の改良パスを省略し、KV キャッシュを 4 分の 1 に縮小します。

## 🎥 オフラインレンダリングパイプライン（`demo_render/batch_demo.py`）

シーケンスがインタラクティブな viser ビューアーには長すぎる場合（例：[上記の屋内ウォークスルー](#-注目屋内ウォークスルー約-25000-フレーム13-分)）は、このパイプラインを使用します。`demo_render/batch_demo.py` は一体型のオフラインエントリーポイントです。動画または画像フォルダーを指定すると、1 つのコマンドでモデル推論を実行し、ヘッドレスの点群フライスルー MP4 を生成します。PyTorch / FlashInfer / チェックポイントのスタックは `demo.py` と共通です。

VRAM 容量や GPU 使用量に制約がある場合は、次の実装も参照できます：https://github.com/ureeey/lingbot-map-rtx4060-8g/commit/eeee84a89cc97c1e39b736b46df4ee315275700b

### インストール（基本インストールの拡張）

**1. レンダリング用 Python 依存関係**

```bash
pip install -e ".[vis,render]"
```

`render` は `open3d>=0.19` と `pyyaml` を導入します（中心となる `numpy<2` 制約は `lingbot-map` の基本インストールに由来します）。このパイプラインの空マスクは、バッチセグメンテーションに `onnxruntime-gpu` を使用します。CPU 版の `onnxruntime` をまだ導入していない場合は、次をインストールしてください。

```bash
pip install onnxruntime-gpu
```

**2. Kaolin** — 上記で推奨した PyTorch 2.8.0 + CUDA 12.8 に対応：

```bash
pip install --index-url https://pypi.org/simple \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

> `--index-url https://pypi.org/simple` は、PyPI のプレースホルダー wheel（インポート時に `ImportError` が発生します）を返す可能性がある内部ミラーを回避します。
> NVIDIA Kaolin は PyTorch 2.9.x 用のビルド済み wheel を公開していません。他の理由で 2.9 を使用する場合は、Kaolin をソースからビルドしてください（`pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git`、ローカル CUDA ツールキットが必要）。その他の torch/CUDA の組み合わせは [NVIDIA Kaolin のインストール](https://kaolin.readthedocs.io/en/latest/notes/installation.html)を参照してください。

**3. ffmpeg**

```bash
sudo apt install ffmpeg    # or: brew install ffmpeg
```

**4. CUDA 拡張**（初回実行前に必須）

```bash
cd demo_render/render_cuda_ext && python setup.py build_ext --inplace && cd ../..
```

これにより、`voxel_morton_ext` と `frustum_cull_ext` がその場でビルドされます。どちらも GPU ボクセル化と視錐台カリングのために `rgbd_render` からインポートされます。

### 実行例 — 長時間の屋内ウォークスルー（約 25,000 フレーム、13 分）

**データセット：**Hugging Face の [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) からサンプル動画をダウンロードします。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/indoor_travel.MP4 \
    --output_folder /data/outputs/indoor_travel/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/indoor.yaml \
    --mode windowed --window_size 128 \
    --keyframe_interval 10 --overlap_keyframes 8 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f4f5e555-22a8-4cc9-b380-dfde5fe1c809" />

各フラグを指定する理由：

| フラグ | 指定する理由 |
|---|---|
| `--mode windowed --window_size 128` | シーケンスが約 320 フレームの RoPE 学習範囲を超えると、スライディングウィンドウ推論が必要になります。各ウィンドウで KV キャッシュがリセットされます。**`window_size` が数えるのは実フレームではなく KV キャッシュのスロットです**。最初の `num_scale_frames`（=8）スロットにはスケールフレームを、残りの `128 − 8 = 120` スロットにはキーフレームを格納します。したがって `keyframe_interval = 13` の場合、1 ウィンドウで `8 + 120 × 13 = 1568` 実フレームをカバーします。 |
| `--keyframe_interval 10` | 10 フレームごとのフレームだけをキーフレームとしてキャッシュします。非キーフレームもフレームごとの予測を出力しますが、KV キャッシュは増加しません。|
| `--overlap_keyframes 8` | 隣接ウィンドウで 8 キーフレーム分のコンテキストを共有します。内部的には `max(num_scale_frames, 8 × keyframe_interval) = 8 × 13 = 104` 実フレームの重複として解決されます。ウィンドウ間の姿勢アラインメントを安定させるため、`keyframe_interval > 1` の場合は常に推奨します。 |
| `--config demo_render/config/indoor.yaml` | 屋内プリセットから、レンダリング、シーン、カメラ、オーバーレイのデフォルト値（短い深度、近接した追従カメラ）を初期化します。ユーザーが明示的に渡した CLI フラグは引き続き YAML 値を上書きします。 |
| `--sky_mask_dir` / `--sky_mask_visualization_dir` | 空マスクとその並列比較画像をディスクへ保存し、以後の再実行で ONNX セグメンテーションをやり直さずに再利用できるようにします。（レンダリングパイプラインが使用するのは、YAML プリセットまたは `--mask_sky` で空マスクを有効にした場合だけです。） |
| `--camera_vis default` | レンダリング動画に軌跡と直近フレームの点を重ねます。 |
| `--keyframes_only_points` | キーフレームの深度だけを点群へ逆投影します。非キーフレームも姿勢を軌跡/視錐台オーバーレイへ提供します。非常に長いシーケンスでも点群を疎に保てます。 |
| `--frame_tag --frame_tag_position top_right` | MP4 の右上へ `<i> / <N> Frames` カウンターを表示します。 |
| `--save_predictions` | フレームごとの NPZ を MP4 とともに保存します。後から確認したり、異なるカメラ/オーバーレイ設定で再レンダリングしたりする場合に便利です。 |


keyframe_interval = 10 を image_stride = 10 に置き換えるとレンダリングを高速化できます。次に、demo_render/config/indoor.yaml のカメラ追従セクションのコメントを解除し、鳥瞰範囲を [2000, 2500] に設定すると、デモに示した屋内フライスルー効果を再現できます。

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/5581d2b2-cb86-4187-a13d-46ac9a22ce99" />





https://github.com/user-attachments/assets/21b444ea-e6b6-48f0-8b34-3acad41166ac







### 実行例 — 屋外走行シーン

**データセット：**Hugging Face の [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) からサンプル動画をダウンロードします。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/drive_frames.mp4 \
    --output_folder /data/outputs/drive/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c26afdb-6bb8-4d20-a7e0-f5a220382662" />


上記の屋内ウォークスルーとの相違点：

| フラグ | 指定する理由 |
|---|---|
| `--config demo_render/config/outdoor_drive.yaml` | 屋外プリセットからデフォルト値を初期化します。空マスクを有効にし、より深いレンダリング範囲（`max_depth: 250`）を使い、車両軌跡向けに調整した追従カメラで最後に鳥瞰表示を行います。 |
| `--image_stride 1` | 動画の全フレームを使用します。長時間または高 FPS の走行動画をサブサンプリングするには値を大きくします。 |
| `--max_non_keyframe_gap 100` | キーフレームを強制するまでの連続非キーフレーム数の上限です。フローベースのキーフレーム選択（`--flow_threshold > 0`）でのみ有効で、デフォルトの固定間隔モードでは効果がありません。 |

残りのフラグ（`--mode windowed --window_size 128`、`--overlap_keyframes 8`、空マスクのキャッシュ、オーバーレイ、`--save_predictions`）は屋内の例と同じです。上記のフラグ別の表を参照してください。

### 実行例 — LingBot-World のシーン

世界モデル LingBot-World で生成された動画を再構成します。同じパイプラインを生成動画にもそのまま使用できます。

**データセット：**Hugging Face の [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) からサンプル動画（`lingbo_world_frames.mp4`、`lingbo_world2_frames.mp4`）をダウンロードします。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/lingbo_world_frames.mp4 \
    --output_folder /data/outputs/lingbo_world/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

2 本目のクリップでは、`--video_path /data/demo_videos/lingbo_world2_frames.mp4 --output_folder /data/outputs/lingbo_world2/` を指定して同じコマンドを実行します（キャッシュしたマスクを分けて保存する場合は、`--sky_mask_dir` / `--sky_mask_visualization_dir` に別々のフォルダーを指定してください）。

すべてのフラグは上記の[屋外走行シーン](#実行例-屋外走行シーン)と同じで、入力動画と出力フォルダーだけが異なります。各フラグの理由は、走行シーンと屋内ウォークスルーの表を参照してください。

<img width="3736" height="1080" alt="image" src="https://github.com/user-attachments/assets/1f60d505-1407-482c-9b5d-57c7145c0b7d" />

<img width="1200" height="339" alt="image" src="https://github.com/user-attachments/assets/e62bedaa-1e61-40b3-8fea-01c8a15355f0" />



### カメラパス（YAML）

仮想カメラパスは、YAML プリセットの `camera.segments` リストで記述します。このプリセットは `--config` で渡します。YAML を編集するだけで独自のショットを設計でき、CLI フラグを変更する必要はありません。

組み込みプリセットは `demo_render/config/` にあります：`default.yaml`、`indoor.yaml`、`outdoor_drive.yaml`。いずれかをコピーし、`camera:` ブロックを編集してください。

#### YAML の構造

```yaml
camera:
  fov: 60.0          # camera field of view in degrees
  transition: 30     # frames blended between adjacent segments
  segments:
    - mode: follow            # chase cam following the input trajectory
      frames: [0, 1500]       # rendered-frame range this segment covers (-1 = end)
      back_offset: 0.3        # how far behind the input camera (fraction of scene scale)
      up_offset: 0.08         # vertical lift above the input camera
      look_offset: 0.4        # how far ahead the lookat target points
      smooth_window: 30       # trajectory smoothing window in frames
    - mode: birdeye           # rise up for a top-down reveal of the whole scene
      frames: [1500, 1800]
      reveal_height_mult: 2.5 # birdeye height = scene scale × this factor
    - mode: follow            # drop back into chase cam
      frames: [1800, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
```

`transition` は、隣接するセグメント間でブレンドするフレーム数を制御します。`frames: [0, -1]` は「シーケンス全体」を意味します。

#### 使用可能なモード

| `mode` | 動作 | 調整可能なフィールド |
|---|---|---|
| `follow` | 追従カメラが滑らかなオフセットで入力軌跡をたどります。ウォークスルーで最も映画的な選択肢です。 | `back_offset`、`up_offset`、`look_offset`、`smooth_window`、`scale_frames` |
| `birdeye` | シーン全体を上から表示します。メインビジュアルや概要ショットに便利です。 | `reveal_height_mult` |
| `static` | eye + lookat を固定し、セグメントの開始フレームから自動導出します。 | — |
| `pivot` | eye を固定し、lookat を軌跡に沿って移動させます。 | — |

#### 単一ショットの YAML 例

**追従のみ**（最も一般的）：

```yaml
camera:
  fov: 60.0
  segments:
    - mode: follow
      frames: [0, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
      smooth_window: 30
```

**全編鳥瞰**（概要/メインビジュアル向け）：

```yaml
camera:
  fov: 60.0
  segments:
    - mode: birdeye
      frames: [0, -1]
      reveal_height_mult: 2.5
```

**追従に鳥瞰を挿入**：`segments:` の下へ複数のセグメントを順番に並べるだけです。隣接セグメントは `transition` フレームを使って補間されます。

> 注意：`--config` で YAML プリセットを読み込むときに、セグメント形状を指定する CLI フラグ（`--camera_mode`、`--back_offset`、`--up_offset`、`--look_offset`、`--smooth_window`、`--follow_scale_frames`、`--birdeye_start`、`--birdeye_duration`、`--reveal_height_mult`）を**1 つでも**渡すと、YAML の `segments` は破棄され、それらのフラグからカメラパスが再構築されます。YAML だけで駆動するには、これらをコマンドラインで渡さないでください。

### 出力ファイル

出力名（例：`<scene>` または `<video_name>`）ごとに、次のファイルが生成されます。

| ファイル | 説明 |
|------|-------------|
| `<name>_pointcloud.mp4` | レンダリングした点群フライスルー |
| `<name>_pointcloud_rgb.mp4` | 元の RGB フレームを動画としてエンコードしたもの |
| `<name>_pointcloud_config.yaml` | この実行の完全な設定スナップショット |
| `batch_results.json` | シーンごとの成功状態/所要時間の要約 |

## 📜 ライセンス

本プロジェクトは Apache License 2.0 のもとで公開されています。詳細は [LICENSE](LICENSE.txt) ファイルを参照してください。

## 📖 引用

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## ✨ 謝辞

有益な議論と支援をいただいた Shangzhan Zhang、Jianyuan Wang、Yudong Jin、Christian Rupprecht、Xun Cao の各氏に感謝します。

本研究は、以下の優れたオープンソースプロジェクトを基盤としています。

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer)

---
