
# Fashion-MNIST Classification with CNN

アドビの期末課題

## 1. プロジェクト概要

Google Colab上で構築した、畳み込みニューラルネットワーク（CNN）による衣類画像の10クラス分類。28x28ピクセルのグレースケール画像データセット「Fashion-MNIST」を使用し、
基本的なCNN構造に、と過学習対策（Dropout）の有無による学習の違いを見る

## 2. アルゴリズムと数式

### 畳み込み演算 (Convolution)

画像の特徴（エッジやパターン）を抽出するために、カーネル（フィルタ）を用いた畳み込み演算の部分
入力画像 $I$ に対してカーネル（フィルタ） $K$とすると

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) K(m, n)$$

### 活性化関数 (ReLU)

モデルに非線形性を導入し、複雑な形状を学習可能にするため、中間層にReLUを使用

$$f(x) = \max(0, x)$$

### 出力層 (Softmax)

最終的な10クラスの予測確率を算出するためにソフトマックス関数を使用

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

## 3. モデル構成
[![](https://mermaid.ink/img/pako:eNp1ktGK1DAUhl8l5LozNG3TdnIhuO10d8BFcfXGdpE4zcwU2qSkiXQd5nbVfYgFwRcQvFHQtxnFtzCTdsYLNRSSw_-dv-ecZAuXomSQwLWk7QY8SwsOzHqYL3irFQFe3JsPXYPJ5AE4yxPBX3spAb4HVlWtmOwc4Pf-9ZB1Zqkkv6T9EyHqiq8PrNd7o55YPT25hMG_XFJLzf_rMrd6lmc1VYpxAlDoukDzSnUjkVniPE8Z75jRvXiQHfCUPXo-QucWushTKVpxaFVSxYA7xaN-YfXFYAIea2Xngdyj1ZVYqYb2hh74Tr8aZpgxqrRkL-e9knSpKsHzn--_7L9__nH3bf_26_7Tx-O4xqmMbY_dDRvj5V_GSU27rlpVS2pN9-9uf324_2OXjX2N5Q_b4mQHHXPHVQmJkpo5sGGyoYcQbg9IAdWGNayAxBxLtqK6VgUs-M6ktZS_EKI5Zkqh15tjoNvSjC2tqCnxRFCtxNUNX54yzO-ZTITmChIUYmsJyRb2JoxmU2-GXW-GMApR4EYOvIFkgqdxEGLsRkaLoihAOwe-sVW403jmY4SDMAoDP_Kj2IGsrJSQl8NTti969xuxBuHU?type=png)](https://mermaid.live/edit#pako:eNp1ktGK1DAUhl8l5LozNG3TdnIhuO10d8BFcfXGdpE4zcwU2qSkiXQd5nbVfYgFwRcQvFHQtxnFtzCTdsYLNRSSw_-dv-ecZAuXomSQwLWk7QY8SwsOzHqYL3irFQFe3JsPXYPJ5AE4yxPBX3spAb4HVlWtmOwc4Pf-9ZB1Zqkkv6T9EyHqiq8PrNd7o55YPT25hMG_XFJLzf_rMrd6lmc1VYpxAlDoukDzSnUjkVniPE8Z75jRvXiQHfCUPXo-QucWushTKVpxaFVSxYA7xaN-YfXFYAIea2Xngdyj1ZVYqYb2hh74Tr8aZpgxqrRkL-e9knSpKsHzn--_7L9__nH3bf_26_7Tx-O4xqmMbY_dDRvj5V_GSU27rlpVS2pN9-9uf324_2OXjX2N5Q_b4mQHHXPHVQmJkpo5sGGyoYcQbg9IAdWGNayAxBxLtqK6VgUs-M6ktZS_EKI5Zkqh15tjoNvSjC2tqCnxRFCtxNUNX54yzO-ZTITmChIUYmsJyRb2JoxmU2-GXW-GMApR4EYOvIFkgqdxEGLsRkaLoihAOwe-sVW403jmY4SDMAoDP_Kj2IGsrJSQl8NTti969xuxBuHU)


| レイヤー | 出力形状 | パラメータ | 説明 |
| --- | --- | --- | --- |
| **Input** | (28, 28, 1) | 0 | 入力データの定義 |
| **Conv2D** | (26, 26, 32) | 320 | 3x3フィルタによる特徴抽出 |
| **MaxPooling2D** | (13, 13, 32) | 0 | 空間情報のダウンサンプリング |
| **Conv2D** | (11, 11, 64) | 18,496 | より高度な形状特徴の抽出 |
| **Flatten** | (1600) | 0 | 1次元ベクトルへの変換 |
| **Dense** | (128) | 204,928 | 特徴の統合 |
| **Dropout(0.5)** | (128) | 0 | 過学習の防止（50%無効化） |
| **Dense(Output)** | (10) | 1,290 | 10クラス分類の出力 |

## 4. 学習の評価
Googlecolabコード上にあるので確認してください

```python

```

## 5. 参考文献

* **Dataset**: [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)


---

## ライセンス

  * このソフトウェアパッケージは，3条項BSDライセンスの下，再頒布および使用が許可されます．
  * © 2025 Takuto Kanno
