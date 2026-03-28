# pytorch-practice

## Overview

このリポジトリは、CIFAR-10 を使って画像分類を学ぶための PyTorch 練習用プロジェクトです。
複数のモデルを個別のスクリプトで試せる構成になっており、主に次の 3 系統を扱っています。

- 自作の CNN
- `torchvision` の ResNet-18
- `transformers` の Vision Transformer (ViT)

パッケージ化された学習フレームワークではなく、実験用スクリプトを並べたシンプルな構成です。

## Models

### 1. Custom CNN

`train_cnn_cifar-10.py` に実装されています。

主な特徴:

- 2 層の畳み込みブロックと Batch Normalization
- Max Pooling によるダウンサンプリング
- 全結合層による分類ヘッド
- Dropout による正則化
- Random Crop、Horizontal Flip、Color Jitter によるデータ拡張

### 2. ResNet-18

`ResNet.py` に実装されています。

主な特徴:

- `torchvision.models.resnet18` を利用
- 最終全結合層を CIFAR-10 の 10 クラス分類用に置き換え
- 検証精度を見ながら最良モデルを保存

### 3. Vision Transformer (ViT)

`ViT.py` に実装されています。

主な特徴:

- Hugging Face Transformers の `ViTForImageClassification` を利用
- CIFAR-10 画像を `224x224` にリサイズして入力
- 10 クラス分類向けの fine-tuning 構成

## Training

各スクリプトはそれぞれ次の処理を完結して持っています。

- データセットの読み込み
- train / validation 分割
- 前処理とデータ拡張
- モデル定義
- 学習ループ
- 検証とテスト評価

データセットは `data_cifar10/` に保存されます。

また、Apple Silicon 環境では MPS を利用し、利用できない場合は CPU にフォールバックします。

## Results

このリポジトリは、厳密な実験管理よりもモデル実装と学習の試行に重きを置いています。

現状確認できる出力は主に次のとおりです。

- 学習中の Training Loss / Validation Loss
- モデル選択のための Validation Accuracy
- 最終的な Test Accuracy
- Custom CNN における誤分類画像の可視化

スクリプト実行により生成されるチェックポイントには次のものがあります。

- `best_resnet18.pth`
- `best_vit.pth`

## How to Run

まず必要なライブラリをインストールします。

```bash
pip install torch torchvision transformers matplotlib
```

環境確認:

```bash
python test.py
```

Custom CNN の学習:

```bash
python train_cnn_cifar-10.py
```

ResNet-18 の学習:

```bash
python ResNet.py
```

ViT の学習:

```bash
python ViT.py
```

## Future Work

- データ読み込みや評価処理の重複を共通化する
- `requirements.txt` または `pyproject.toml` を追加する
- 損失や精度のログを継続的に記録できるようにする
- モデルごとの結果を一覧で比較できる形に整理する
- チェックポイント保存と再現性の扱いを改善する
- ViT の学習フローを安定させて結果を確認できるようにする
- Google Colab で学習しやすい構成を整える
