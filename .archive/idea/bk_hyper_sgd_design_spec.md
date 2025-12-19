# BK-HyperSGD: Trust-Region-Clipped Riemannian Momentum Optimizer

## 概要
BK-HyperSGD は、**双曲空間（ポアンカレ球）**上での学習を安定化させるために設計された、**Trust Region（信頼領域）ベースのリーマン最適化アルゴリズム**である。

従来の SGD / Adam が「学習率」というスカラー値のみで更新幅を制御するのに対し、BK-HyperSGD は **リーマン幾何に基づく距離（計量）**を用いて、1 ステップあたりの移動量を**物理的に制限**する。

その結果、以下の双曲空間特有の不安定性を構造的に防止する。

- **特異点付近での勾配爆発**（相転移点・急峻な損失崖）
- **ポアンカレ球境界への吸い込み・NaN 化**
- **モーメンタムの幾何的不整合（接空間のズレ）**

本オプティマイザは、BK-Core / ResNet-BK のような双曲表現モデルの「足回り」を支えることを目的としている。

---

## 幾何学的前提

### 多様体定義（曲率 −c, c > 0）

\[
\mathbb{B}_c^n = \{ x \in \mathbb{R}^n \mid c\|x\|^2 < 1 \}, \quad \|x\| < 1/\sqrt{c}
\]

### 共形計量

ポアンカレ球上の計量は共形であり、以下で与えられる。

\[
\lambda_x = \frac{2}{1 - c\|x\|^2}
\]

\[
g_x = \lambda_x^2 I
\]

接空間における内積およびノルムは：

\[
\langle u, v \rangle_{g_x} = \lambda_x^2 \langle u, v \rangle
\]

\[
\|u\|_g = \lambda_x \|u\|_2
\]

境界に近づくほど \( \lambda_x \) は発散し、更新は幾何学的に厳しく制限される。

---

## リーマン勾配

ユークリッド勾配 \( g_E = \nabla f(x) \) から、リーマン勾配 \( g_R \) は以下で与えられる。

\[
 g_R = \mathrm{grad}\, f(x) = \frac{1}{\lambda_x^2} g_E
\]

これにより、**双曲幾何に整合した最急降下方向**が得られる。

---

## アルゴリズム概要

### ハイパーパラメータ

- 学習率：\( \eta \)（最大速度）
- Trust Radius：\( \Delta \)（1 ステップの最大移動距離）
- モーメンタム係数：\( \beta \)
- 数値安定化定数：\( \varepsilon \)

---

## BK-HyperSGD ステップ

### 1. 共形係数（数値安定版）

\[
s = 1 - c\|x\|^2
\]

\[
s \leftarrow \max(s, \varepsilon)
\]

\[
\lambda_x = 2 / s
\]

---

### 2. リーマン勾配計算

\[
 g_R = g_E / \lambda_x^2
\]

---

### 3. モーメンタム（接空間の運搬付き）

接空間は点ごとに異なるため、前ステップのモーメンタムは **vector transport** により現在位置へ運搬する。

\[
 \tilde{m}_{t-1} = \mathrm{Transp}_{x_{t-1} \to x_t}(m_{t-1})
\]

\[
 m_t = \beta \tilde{m}_{t-1} + (1 - \beta) g_R
\]

---

### 4. Trust Region Clipping（リーマンノルム）

\[
 \text{step}_{\mathrm{riem}} = \eta \|m_t\|_g = \eta \lambda_x \|m_t\|_2
\]

\[
 \alpha = \min\left(1, \frac{\Delta}{\text{step}_{\mathrm{riem}}} \right)
\]

\[
 m_t \leftarrow \alpha m_t
\]

これにより、

\[
 \eta \|m_t\|_g \le \Delta
\]

が常に保証される。

---

### 5. 多様体更新（Retract / ExpMap）

接空間での更新：

\[
 u = -\eta m_t
\]

多様体へ戻す：

\[
 x_{t+1} = \mathrm{Retr}_{x_t}(u)
\]

（理想は ExpMap、実装上は Retract 近似で可）

---

### 6. Boundary Projection（最終安全装置）

\[
 \|x_{t+1}\| \le (1 - \varepsilon) / \sqrt{c}
\]

を満たすよう、球内へ射影する。

---

## 数値安定性・実装上の注意

- \( \lambda_x \) 計算時は必ず \( \varepsilon \) で下限を設ける
- 勾配・モーメンタムに NaN / Inf が含まれる場合はステップをスキップ
- 双曲パラメータは **float64 推奨**（境界付近での精度確保）
- ログ推奨指標：
  - max(\|x\|)
  - max(\lambda_x)
  - mean(step_riem)
  - clipping rate

---

## 位置づけ

BK-HyperSGD は、単なる Gradient Clipping ではなく、

> **「双曲空間上で許容される物理的移動距離」を直接制御する最適化器**

である。

これにより、

- 特異点でも滑落しない
- 境界に吸い込まれない
- モーメンタムが幾何的に破綻しない

という制約を同時に満たし、BK-Core の安定学習を根本から支える。

