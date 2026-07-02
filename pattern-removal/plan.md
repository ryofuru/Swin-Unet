# pattern-removal 学習プロジェクト 計画

Swin-Unet を利用して、投影パターンを動画フレームから除去するモデルを学習する。
gapgrid プロジェクトとは独立した、このサブフォルダに閉じたプロジェクトとする。

## タスク

- 入力: パターンが投影された動画フレーム(カラー)
- 出力: パターンを除去した動画フレーム(カラー)
- 入力・出力ともにカラー画像(RGB)。

## 学習データ

場所:
```
/home/lab-shared/gitrep/blender-render-tool/_test/medshape-colon-shapes-sequence-output-dist/
  sequence_0000/proj_pat0/   (入力, 39フレーム, 0000.png-0038.png)
  sequence_0000/pointlight/  (GT,   39フレーム)
  sequence_0001/proj_pat0/   (入力, 32フレーム)
  sequence_0001/pointlight/  (GT,   32フレーム)
  sequence_0002/proj_pat0/   (入力, 17フレーム)
  sequence_0002/pointlight/  (GT,   17フレーム)
```
- 各画像: 512x512 RGB PNG。
- `proj_pat0` フォルダ内の連番画像が入力動画のフレーム、`pointlight` フォルダ内の連番画像がそれに対応するGT動画のフレーム。
- `sequence_XXXX` ごとに1本の連続した動画。フレーム番号は動画内の時刻に対応する。
- 全体で3本の動画・合計88フレームのみ。小規模データセットであることに注意。

## 動画の扱い方(8フレーム = 1動画単位)

Swin-Unet 本体(`networks/swin_transformer_unet_skip_expand_decoder_sys.py`)は2次元画像専用のアーキテクチャであり、時間方向のAttentionは存在しない。そのため、時間方向の8フレームをチャンネル方向に連結して2次元画像として扱う(gapgridがIN_CHANS=4を使っている前例を踏襲)。

- 1動画単位 = 連続する8フレーム(`NUM_FRAMES = 8`)
- 入力チャンネル数 = 出力チャンネル数 = 8フレーム × RGB3チャンネル = 24
- 入力テンソル: `proj_pat0` の連続8フレームをチャンネル軸で連結 → `(B, 24, H, W)`
- 出力テンソル: モデルが出力する24チャンネルを8枚のRGBフレームに復元し、`pointlight` の対応する8フレームと比較する。

## クリップの切り出し方・train/val分割

各シーケンス内でスライディングウィンドウ(stride=1)により8フレームのクリップを生成する。

- sequence_0000(39フレーム) → 32クリップ
- sequence_0001(32フレーム) → 25クリップ
- 上記2本(計57クリップ)を **train** に使用
- sequence_0002(17フレーム) → 10クリップを **val** に使用(動画単位で分離し、train/val間のフレーム重複によるリークを避ける)

## 画像サイズ(クロップ)

- 学習データの512x512から **256x256** を切り出して使用する。
- train: クリップ内で1回だけランダムな切り出し位置・左右反転を決定し、8フレーム全ておよび入力・GT両方に同一の処理を適用する(フレーム間で切り出し位置がずれないようにする)。
- val: 中央切り出し、反転なし。
- 画素値は float32 [0,1] に正規化(255で除算)。

## バッチサイズ

- バッチサイズ = 8(8クリップ/バッチ)

## モデル設定上の注意

- `WINDOW_SIZE` は画像サイズ256に対応する値(8)を使う。既存configのWINDOW_SIZE=7は224専用で、256では `window_partition` の reshape に失敗するため使用不可。
- `config.DATA.IMG_SIZE` を256に明示的に設定する必要がある(`--img_size` CLI引数だけでは反映されない)。
- 学習済みSwin-T重み(`pretrained_ckpt/swin_tiny_patch4_window7_224.pth`)はエンコーダ・デコーダの深い層には流用可能だが、入力チャンネル数が3→24に変わるため、`patch_embed.proj.weight` の拡張方法を別途検討する(8フレーム分にRGB重みをタイル状に複製する案)。

## 損失・出力の扱い

- 出力ヘッドは活性化なしのConv2dのため、学習時にsigmoidを適用して[0,1]に収める。
- 損失関数は L1Loss を予定(画像復元タスクでの標準的選択)。
- オプティマイザは Adam を予定(セグメンテーション用のSGD+poly減衰ではなく、復元タスクに適した設定)。

## ファイル構成(予定、未実装)

```
pattern-removal/
  plan.md                                  (本ファイル)
  dataset_pattern_removal.py               データセットクラス
  trainer_pattern_removal.py               学習ループ
  train_pattern_removal.py                 CLIエントリポイント
  configs/swin_tiny_patch4_window7_224_patternremoval.yaml
  train_pattern_removal.sh
```
上記のコード一式は本計画確定後の次のステップで実装する。
