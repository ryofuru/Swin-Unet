# Channel-Swin-Unet 実装計画 (案A)

作成日: 2026-06-27

## 概要

Swin-Unetの派生モデル。Channel-ViTの考え方を導入し、入力4チャネル（h_phase, v_phase, colored_G, traindata_code）それぞれに独立したトークン列を割り当て、Swinのウィンドウ内で空間次元とチャネル次元をまとめてAttentionする。

## ファイル構成

```
networks/channel_swin_transformer_unet.py   ← 新モデル（全クラス）
configs/channel_swin_tiny_patch4_window7_224_gapgrid.yaml
train_gapgrid_channelswin.py                ← 学習スクリプト
train_gapgrid_channelswin.sh                ← 学習シェルスクリプト
trainer_gapgrid_channelswin.py              ← Trainerロジック
infer_gapgrid_channelswin.py                ← 推論スクリプト
debug_infer_center_channelswin.py           ← デバッグ可視化スクリプト
```

## 入出力仕様

- 入力: `(B, 4, H, W)` float32 — 4チャネル画像（変更なし）
- 出力: `(B, NUM_CLASSES, H, W)` — ピクセル単位クラス分類（変更なし）
- `NUM_CLASSES = 2389`、`IGNORE_INDEX = -1`（変更なし）

## アーキテクチャの変更点

### トークン表現の変化

| | 現行 Swin-Unet | Channel-Swin-Unet |
|---|---|---|
| PatchEmbed | `Conv2d(4, D, patch=4)` → `(B, H'W', D)` | `[Conv2d(1, D, patch=4)] × 4` → `(B, H'W', 4, D)` |
| ウィンドウ内トークン数 | `Wh×Ww = 49` | `Wh×Ww×C_in = 196` |
| Attention コスト | `O(49²)` | `O(196²)` ≈ ×16 |
| 内部テンソル形状 | `(B, H'W', D)` | `(B, H'W', C_in, D)` または `(B, H'W'×C_in, D)` |

### 実装するクラス一覧（`channel_swin_transformer_unet.py`）

#### 1. `ChannelPatchEmbed`
```
入力: (B, 4, H, W)
処理:
  - 4つの独立した Conv2d(1, D, patch_size, patch_size)（per-channel projection）
  - 各チャネルのトークン列に学習可能チャネル位置埋め込みを加算
      channel_embed: nn.Parameter(torch.zeros(C_in, D))  → trunc_normal_ 初期化
  - Norm後に (B, H'W', C_in, D) として返す
出力: (B, H'W', C_in, D)
```

#### 2. `ChannelWindowAttention`
```
入力: (num_windows×B, Wh×Ww×C_in, D)
処理:
  - QKV projection: Linear(D, 3D)
  - 3D 相対位置バイアス（分解表現）:
      bias(dh, dw, dc) = spatial_bias[dh, dw] + channel_bias[dc]
      spatial_bias:  (2Wh-1)×(2Ww-1), num_heads  ← 現行のまま流用可
      channel_bias:  (2×C_in-1, num_heads)         ← 新規追加
  - Scaled dot-product attention on N = Wh×Ww×C_in tokens
出力: (num_windows×B, Wh×Ww×C_in, D)
```

#### 3. `ChannelSwinTransformerBlock`
```
入力: (B, H', W', C_in, D)
処理:
  - シフト: 空間次元のみ (Wh//2, Ww//2)、チャネル次元はシフトなし
  - ウィンドウ分割: (B, H', W', C_in, D) → (num_windows×B, Wh×Ww×C_in, D)
  - ChannelWindowAttention → ウィンドウ再構成
  - 空間マスク: 空間シフト分のみ（チャネル間は常に全接続）
  - FFN (MLP): D → 4D → D（変更なし）
出力: (B, H', W', C_in, D)
```

#### 4. `ChannelPatchMerging`（ダウンサンプル）
```
入力: (B, H'×W'×C_in, D)  ← L = H'×W', C = C_in
処理:
  - reshape: (B, H', W', C_in, D)
  - 2×2 空間パッチを連結: (B, H'/2, W'/2, C_in, 4D)
  - flatten to: (B, H'/2×W'/2×C_in, 4D)
  - Linear(4D, 2D) + Norm
  - チャネル次元 C_in は保持
出力: (B, H'/2×W'/2×C_in, 2D)
```

#### 5. `ChannelPatchExpand`（アップサンプル ×2）
```
入力: (B, H'×W'×C_in, D)
処理:
  - Linear(D, 4×D/2) = Linear(D, 2D)
  - reshape: (B, H', W', C_in, 2, 2, D/2) → (B, 2H', 2W', C_in, D/2)
  - flatten spatial: (B, 2H'×2W'×C_in, D/2)
出力: (B, 2H'×2W'×C_in, D/2)
```

#### 6. `ChannelFinalPatchExpand_X4`（最終 ×4 アップサンプル）
```
入力: (B, H'×W'×C_in, D)
処理:
  - Linear(D, 16×D)
  - reshape: (B, H', W', C_in, 4, 4, D) → (B, 4H', 4W', C_in, D)
  - チャネル集約: reshape to (B, 4H', 4W', C_in×D) → Linear(C_in×D, D)
    （または平均: (B, 4H', 4W', D) に mean over C_in dim）
  - reshape: (B, D, 4H', 4W') for 最終 Conv2d
出力: (B, D, H, W)  ← 元の空間解像度
最終: Conv2d(D, NUM_CLASSES, 1) でクラスロジット
```

#### 7. `ChannelBasicLayer` / `ChannelBasicLayer_up`
```
- N × ChannelSwinTransformerBlock のスタック
- ChannelBasicLayer: 末尾で ChannelPatchMerging（最終ステージを除く）
- ChannelBasicLayer_up: 先頭で ChannelPatchExpand
- Skip connection の concat_back_dim:
    Linear(2×C_in×D, C_in×D) で Encoder skip + Decoder を結合
```

#### 8. `ChannelSwinTransformerSys`
```
Encoder (4ステージ):
  ChannelPatchEmbed → 
  ChannelBasicLayer[0] (depth=2, D=96) → x_down[0] →
  ChannelBasicLayer[1] (depth=2, D=192) → x_down[1] →
  ChannelBasicLayer[2] (depth=6, D=384) → x_down[2] →
  ChannelBasicLayer[3] (depth=2, D=768) [bottleneck]

Decoder (4ステージ):
  ChannelBasicLayer_up[0] + x_down[2] skip →
  ChannelBasicLayer_up[1] + x_down[1] skip →
  ChannelBasicLayer_up[2] + x_down[0] skip →
  ChannelBasicLayer_up[3] →
  ChannelFinalPatchExpand_X4 →
  Conv2d(D, NUM_CLASSES, 1)
```

#### 9. `ChannelSwinUnet`（エントリポイント）
```
- SwinUnet の代替（vision_transformer.py は使わない）
- grayscale → RGB repeat は不要（4チャネルをそのまま受け取る）
- load_from(): Swin-T 事前学習重みは直接転用不可
    → per-channel Conv 初期化: 元の Conv2d(3,D,...) の重みを列方向に分割して流用
    → それ以外（SwinBlock重み等）: 形状不一致のためスクラッチ
```

## 設定ファイル

`configs/channel_swin_tiny_patch4_window7_224_gapgrid.yaml`:
```yaml
MODEL:
  TYPE: channel_swin
  NAME: channel_swin_tiny_patch4_window7_224
  DROP_RATE: 0.0
  DROP_PATH_RATE: 0.1
  SWIN:
    EMBED_DIM: 96
    DEPTHS: [2, 2, 6, 2]
    DECODER_DEPTHS: [2, 2, 2, 2]
    NUM_HEADS: [3, 6, 12, 24]
    WINDOW_SIZE: 7
    IN_CHANS: 4
    MLP_RATIO: 4.
    QKV_BIAS: True
    QK_SCALE: ~
    APE: False
    PATCH_NORM: True
```

注意: `WINDOW_SIZE: 5` にするとウィンドウトークン数 `5×5×4=100`（`7×7=49` の約2倍）に抑えられる。

## 学習スクリプト

### `train_gapgrid_channelswin.py`
- `train_gapgrid.py` をベースに以下を変更:
  - `from networks.channel_swin_transformer_unet import ChannelSwinUnet` に変更
  - `SwinUnet` → `ChannelSwinUnet`
  - `--pretrained` フラグの挙動を変更（Swin-T重みの部分的転用のみ）

### `train_gapgrid_channelswin.sh`
- `train_gapgrid.sh` と同様の構造
- デフォルト設定: `WINDOW_SIZE=7` または `WINDOW_SIZE=5`（メモリ上限に応じて）

### `trainer_gapgrid_channelswin.py`
- `trainer_gapgrid.py` と基本的に同一（損失関数・可視化ロジックは変更なし）
- 必要なら後でマージ可

## 推論スクリプト

### `infer_gapgrid_channelswin.py`
- `infer_gapgrid.py` をベースに以下を変更:
  - `ChannelSwinUnet` をロード
  - 出力先: `model_out/gapgrid_channelswin/`

### `debug_infer_center_channelswin.py`
- `debug_infer_center.py` をベースに `ChannelSwinUnet` に変更

## コスト・トレードオフ

| 項目 | 現行 Swin-Unet | Channel-Swin-Unet |
|---|---|---|
| ウィンドウトークン数 (W=7) | 49 | 196 |
| Attention FLOPs | 基準 | ×16（同 window size の場合）|
| Window size=5 での FLOPs | — | ×(100/49)² ≈ ×4 |
| パラメータ数増加 | — | PatchEmbed の 4 proj + channel_embed 分のみ（小さい）|
| 事前学習 | Swin-T 利用可 | 部分的（Swin Block 重みは流用不可）|
| チャネル間情報混合 | PatchEmbed 時のみ | 全ステージで Attention により混合 |

## 実装順序（推奨）

1. `ChannelPatchEmbed`
2. `ChannelWindowAttention`（位置バイアスの分解表現）
3. `ChannelSwinTransformerBlock`（ウィンドウ分割・再構成）
4. `ChannelPatchMerging` / `ChannelPatchExpand`
5. `ChannelFinalPatchExpand_X4`（チャネル集約）
6. `ChannelBasicLayer` / `ChannelBasicLayer_up`
7. `ChannelSwinTransformerSys` + `ChannelSwinUnet`
8. 学習・推論スクリプト整備
9. 動作確認: forward pass shape check → 短期学習 → infer 可視化
