# MUSE: Local Training & Optimization Guide (Phase 8)

このドキュメントは、MUSE (ResNet-BK) Phase 8 モデルをローカル環境（特にRTX 3080等のコンシューマGPU）で育成し、その性能を極限まで引き出すためのロードマップです。

---

## 🚀 1. クイックスタート (1Bモデルでの動作確認)

まずは小規模な10億（1B）モデルで、環境と圧縮メカニズムが正常に動作するか確認します。

1.  **データの準備**
    ```bash
    make data-lite
    ```
    *   最低限の学習データをダウンロードします。

2.  **レシピの作成**
    ```bash
    make recipe
    ```
    *   ウィザードに従い、「Balanced」などを選択します。

3.  **1Bモデルの圧縮と初期化**
    ```bash
    make compress-1b
    ```
    *   10億パラメータのモデルを作成し、QHTT (INT8) 形式で `checkpoints/compressed_1b_start/` に保存します。
    *   サイズが数百MB〜1GB程度になっていることを確認してください。

4.  **訓練開始**
    ```bash
    make train-compressed
    ```
    *   学習が開始され、Lossが下がっていくことを確認します。
    *   **チェックポイント:** 100ステップ程度進み、エラー落ちしなければ成功です。

---

## 🌌 2. 本番学習 (10Bモデルへの挑戦)

動作確認ができたら、本命の100億（10B）モデルの学習に進みます。

1.  **全データの準備**
    ```bash
    make data
    make data-ja  # 日本語能力を強化する場合
    ```

2.  **10Bモデルの圧縮と初期化**
    ```bash
    make compress-10b
    ```
    *   **魔法の瞬間:** 100億パラメータ（通常37GB）が、約2.3GBに圧縮されて生成されます。

3.  **訓練開始**
    ```bash
    make train-10b
    ```
    *   RTX 3080 (8GB VRAM) でも動作するように調整されています。
    *   **所要時間:** 数日〜1週間程度で、基礎的な知能が芽生えます。

---

## ⚡ 3. さらなる高速化 (Triton Optimization)

ローカル環境での学習効率をさらに上げるための、エンジニアリング的な改善点です。

### A. BitNetカーネルの統合
*   **現状:** 1.58bit化 (`use_bitnet: true`) はPyTorchで行っています。
*   **Next Step:** これをTritonカーネル (`src/kernels/`) で書き直すことで、計算速度を **2〜3倍** 高速化できます。
*   **対象ファイル:** `src/models/semiseparable_matrix.py` 内のforward処理。

### B. BK-Coreの融合 (Fused Kernel)
*   **現状:** アテンションの代替であるBK-Coreは、PyTorchのJITコンパイルに頼っています。
*   **Next Step:** `src/kernels/bk_scan.py` をさらに最適化し、メモリ読み書きを最小限にする「Fused Kernel」を完成させることで、長い文脈（Long Context）での速度低下を防げます。

---

## 🛡️ 4. 学習の安定化テクニック

圧縮モデルは「軽量」ですが、その分「繊細」です。学習が不安定になった場合の対処法です。

1.  **Lossが NaN になる場合**
    *   `configs/phase8_10b.yaml` の `learning_rate` を **1e-4** から **1e-5** に下げる。
    *   `use_mixed_precision` を `false` にして（速度は落ちますが）FP32で計算してみる。

2.  **Lossが下がらない場合**
    *   `gradient_accumulation_steps` を **32** から **64** に増やし、擬似的なバッチサイズを大きくする。
    *   `configs/dataset_mixing.yaml` で「教科書データ（Cosmopedia）」の比率を増やす。

---

## 🔮 5. 未来への展望

このモデルが完成した暁には、以下のことが可能になります。

*   **自分だけのGPT:** クラウドに依存せず、あなたのPCの中で、あなたの好みの知識を持ったAIが思考します。
*   **エッジデバイスへの移植:** 2.3GBなら、スマホやノートPC、あるいはRaspberry Piのようなデバイスでも（推論なら）動作する可能性があります。

**Enjoy the journey of creating intelligence!**
