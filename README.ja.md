# mimetic-a0 for ComfyUI

[English](README.md)

[mimetic-a0](https://huggingface.co/micknoisemachine/mimetic-a0)（Mick Grierson、MIT）で、テキストから音楽を作るノードです。
学習データは、ライセンスのはっきりした音だけです（Freesound の CC0 のクリップと、作者自身のジェネレーターで作った合成音楽）。
48 kHz ステレオで最長 16 秒。ドラムループ、ベースライン、コード、メロディー、パッド、ワンショット、フルミックスを、指定した
テンポ・キー・コード進行で生成します。

モデルは ComfyUI 自身の stable-audio のクラス（DiT、オートエンコーダー、T5-base テキストエンコーダー）で動かし、公式ツールと同じ
方法でサンプリングします。ComfyUI 以外の Python パッケージは不要で、ワークフローの実行中に何かをダウンロードすることはありません。

非公式の統合です。モデルの作者、Stability AI、Comfy Org とは関係ありません。

## ノード

| ノード | 内容 |
| --- | --- |
| **mimetic-a0 Loader** | `models/mimetic_a0/<name>/` を読み込み（fp32 2 GB または fp16 1 GB）、MODEL と VAE を出力します。 |
| **mimetic-a0 Prompt (BPM and bars)** | コマンドラインの `--bpm` / `--bars` と同じです。プロンプトに「90 BPM, 4 bars」がなければ追加し、長さをちょうどその小節数（4/4）にします。 |
| **mimetic-a0 Sampler** | 公式のサンプラーです。長さ + 6 秒の余白、公式と同じ初期ノイズ、log-SNR で等間隔の 50 ステップ（Euler）、ゼロにしたプロンプトに対する CFG 6 と adaptive projected guidance。1 回のバッチで複数のテイクを作れます。 |
| **mimetic-a0 Decode** | テイクごとにデコードし、長さに合わせて切り詰めます。既定では、コマンドラインの WAV と同じく各テイクのピークを -1 dBFS にします。 |

プロンプトは、コアの **Load CLIP**（`t5_base.safetensors`、type `stable_audio`）と **CLIP Text Encode** に通します。

![text_to_music ワークフロー](docs/images/text_to_music.png)

## インストール

ComfyUI Manager か Comfy Registry（`mimetic-a0-music`）から、または:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-MimeticA0
```

続いてモデルファイルを取得します（リビジョン固定、全ファイルを SHA-256 で確認）:

```bash
python custom_nodes/ComfyUI-MimeticA0/tools/download_model.py --models-dir models            # fp32 の重み + T5-base、2.9 GB
python custom_nodes/ComfyUI-MimeticA0/tools/download_model.py --models-dir models --weights all   # fp16 の重みも
```

`models/mimetic_a0/mimetic-a0/`（model_config.json、重み、リリースの LICENSE・NOTICE・README・PROMPTING.md）と、ComfyUI の
Stable Audio ワークフローが使う T5-base のファイル `models/text_encoders/t5_base.safetensors` ができます。すでに
`t5_base.safetensors` がある場合は、そのまま残します。ComfyUI を再起動し、`workflows/gui/` のワークフローを開いてください:

- `text_to_music.json`: 90 BPM・4 小節（10.7 秒）のブーンバップのループを 1 テイク。
- `four_takes.json`: fp16 の重みで、チェロの単音を 4 テイク。

## プロンプト

モデルは学習時のキャプションの言い回しに最もよく従います。モデルフォルダーの `PROMPTING.md` に、知っている楽器・スタイル・
句がすべて載っています。何の音か、次に音楽の内容、最後にテンポ・キー・長さの順に書きます:

```
hip hop drum loop, standard kit, swung, with ghost notes, 90 BPM, 4 bars
finger bass bassline, over Cm - Fm - G7, sparse, 95 BPM, C minor, 4 bars
electric piano chords, chords Am7 - D9 - Gmaj7, smooth, sustained, 100 BPM, G major, 2 bars
808 kit snare hit, one-shot, hard
single cello note, G3
```

長さは指定した小節数に合わせてください（Prompt ノードが計算します）。50 ステップと CFG 6 は、モデルが検証された設定です。
シードで結果が変わるので、複数のテイクを作ってください。モデルの README によると、得意なのは音楽で（効果音やフィールド録音は
弱い）、4/4 のみ、1・2・4 小節、最長 16 秒です。楽器の多くは General MIDI の音色のように聞こえ、オートエンコーダーが最高音域や
ノイズの質感の上限を決めます。

## 公式ツールと同じか

公式コード（mimetic-a0 9f91b67、stable-audio-tools 3241adb、torch 2.7.1）と RTX 5090 で 7 ケースを比べました。詳細は
[docs/VERIFICATION.md](docs/VERIFICATION.md) にあります。

- **モデル・サンプラー・デコーダー**: 同じテキスト埋め込みを与えると、fp32 のテイクは公式と 95〜116 dB で一致します（最大差は
  フルスケールの 3e-5）。異なる PyTorch ビルド間の丸め誤差の水準です。初期ノイズ、スケジュール、条件付けはビット一致です。
- **テキストエンコーダー**: ComfyUI は T5-base を fp32 で計算し、公式コードは fp16 で計算します。埋め込みの差は 49〜59 dB で、
  ガイダンス付きの 50 ステップを経ると、公式との差は 32〜89 dB になります（同じ音楽で細部が少し違う程度）。公式の fp16 の埋め込み
  自体も安定していません。同じ公式の手順でも、transformers を 4.57.6 から 5.20.1 に変えると 56 dB ずれます。
- **fp16 の重み**: 公式コードはサンプラーの状態も fp16 で保持し、ComfyUI は fp32 で保持します。テイクの差は 36 dB です。公式の
  fp16 と fp32 のテイクも同じシードで 19 dB 離れているので、モデルの README にあるとおり、fp16 では同じシードでも別のテイクになります。
- **精度**: 公式のサンプラーは TF32 を切ります。PyTorch の既定（cuDNN の畳み込みが TF32）のままだと、潜在表現は公式から
  110〜115 dB ではなく 61〜64 dB 離れました。そのため Sampler と Decode ノードは、実行中だけ TF32 を切り、終わったら ComfyUI の
  設定に戻します。
- **シード**: `noise = official` は公式コードと同じく、サンプリングするデバイス上で初期ノイズを作ります。`cpu` は CPU 上で作るので、
  どのマシンでも同じになります。

RTX 5090 では、10.7 秒のテイクが ComfyUI のキュー経由で 1.6 秒です（モデルを読み込む初回は 3.2 秒）。fp32 の重みと T5-base を
読み込んだ状態で、PyTorch が確保した VRAM は最大 3.8 GB でした。CPU でも動きますが、遅くなります。

## ライセンス

このパックのコードは MIT です（[LICENSE](LICENSE)）。モデルは MIT、T5-base は Apache-2.0 です。モデルの NOTICE にある学習データの
クレジットは [NOTICE](NOTICE) にまとめました。その中の GeneralUser GS サウンドフォントについては、一部のサンプルの出所が不明だと
作者自身が書いています。
