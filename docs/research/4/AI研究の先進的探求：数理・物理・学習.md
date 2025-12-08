# **幾何学的知能の統一理論：リーマン最適化、量子化、および力学系による次世代ニューラルアーキテクチャの研究報告書**

## **エグゼクティブサマリー**

本報告書は、現代のディープラーニングが直面している「ユークリッド空間の限界」と「計算効率の物理的制約」という二重の課題に対し、リーマン幾何学、量子化理論、およびKoopman作用素理論を統合した新たなアーキテクチャの数学的・アルゴリズム的基盤を包括的に論じるものである。

大規模言語モデル（LLM）や生成AIのパラメータ数が爆発的に増加する中、1.58bit（3値）のような極低精度量子化（BitNet等）への移行は不可避となっている。しかし、言語や知識の階層構造（ヒエラルキー）を自然に表現できる「双曲空間（Hyperbolic Space）」と、離散的な「量子化（Quantization）」を数学的に矛盾なく統合することは極めて困難である。本研究では、この未踏の領域に対し、以下の5つの柱に基づく解決策を提示する。

1. **リーマン多様体上の量子化最適化**: 負の定曲率空間における被覆数（Covering Numbers）の増大則に基づき、Fisher情報行列を用いた感度認識型量子化（RSAVQ）と、量子化誤差を許容する勾配追跡法（Q-RGT）を融合させる。  
2. **双曲Muonオプティマイザ**: 従来の直交勾配法をLorentz群 $O(n,1)$ 上へ拡張し、Higham-Schulz反復を用いた一般化極分解（Generalized Polar Decomposition）による「J-直交性」の保存手法を確立する。  
3. **Lorentz Batch Normalization**: ノルム（階層深度）を破壊せずに共変量シフトを防ぐため、Lorentz中心（Fréchet mean）と平行移動（Parallel Transport）を用いた正規化手法を採用する。  
4. **物理法則に基づく誤差逆伝播**: 測地線誤差逆伝播（Geodesic Backprop）による勾配流の物理的解釈と、確率共鳴（Stochastic Resonance）を応用した確率的丸め（Stochastic Rounding）による微小勾配の復元メカニズムを解明する。  
5. **Koopmanスペクトル安定性とメタ学習**: リカレントなダイナミクスの安定化を図るEigenloss正則化と、多様体の曲率や学習率テンソルを動的に制御するMeta-SGD/Self-Tuning Networksの導入。

本稿は、単なる既存手法の羅列ではなく、これらを有機的に結合させた「物理幾何学的AI（Physico-Geometric AI）」の設計図である。

## ---

**第1章 序論：幾何学的不整合と効率性のジレンマ**

### **1.1 背景：ユークリッド空間の呪縛と1.58bitの衝撃**

現在の主流なニューラルネットワークは、重みや活性化関数をユークリッド空間 $\\mathbb{R}^n$ 上のベクトルとして扱う。しかし、自然言語の構文木、知識グラフ、生物学的系統樹など、実世界のデータの多くは階層的（Hierarchical）な構造を持っており、これらはユークリッド空間ではなく、負の曲率を持つ双曲空間（Hyperbolic Space）に埋め込むことが幾何学的に最適であることが知られている 1。ユークリッド空間でこれらを表現しようとすると、次元数やパラメータ数を指数関数的に増やす必要が生じ、これがモデルの肥大化を招いている。

一方で、計算コストとメモリ帯域の制約から、BitNet b1.58に代表されるような、重みを $\\{-1, 0, 1\\}$ の3値（1.58bit）に量子化する試みが成功を収めている 3。これは「行列積（Mul）」を「加算（Add）」に置き換える革命的な手法だが、離散値を取るパラメータは微分不可能であり、従来のバックプロパゲーションとの相性が悪い。

### **1.2 本研究の目的：多様体×量子化の統合**

本研究の核心的な問いは、「曲がった空間（多様体）の上で、離散的なパラメータ（量子化）をどのように最適化するか」である。多様体上では「直線」が存在しないため、従来の量子化グリッドや勾配更新則は破綻する。本報告書では、リーマン幾何学の厳密な枠組みの中で、量子化誤差を「多様体上のノイズ」として扱い、それを確率共鳴によって信号として活用する新たな理論体系を構築する。

## ---

**第2章 リーマン多様体上の量子化最適化：数学的アプローチ**

### **2.1 負の定曲率空間における離散性の課題**

双曲空間（ポアンカレ球やLorentzモデル）における最大の特徴は、原点から離れるにつれて空間の体積が指数関数的に増大することである。これは、ユークリッド空間における多項式的な増大とは決定的に異なる 4。

#### **2.1.1 被覆数（Covering Numbers）と量子化誤差**

多様体上の確率測度を有限個の点（量子化された重み）で近似する場合、その近似誤差（量子化誤差）の減衰率は、多様体の幾何学的性質に依存する。文献 4 によれば、リーマン多様体上の量子化誤差の漸近挙動は、球の被覆数（Covering Numbers）の成長率に関連している。  
負の断面曲率を持つ双曲空間では、被覆数が指数関数的に成長するため、原点付近（階層の上位概念）では少ないビット数で十分な表現が可能だが、境界付近（階層の下位概念、葉ノード）では、ユークリッド空間と同じビット数では空間を「埋め尽くす（Cover）」ことができず、量子化誤差（Voronoiセルの歪み）が極端に大きくなる。  
**洞察:** 1.58bitのような極低精度量子化を一様に適用すると、双曲空間の「広がり」に対応できず、階層の深い部分での表現力が崩壊するリスクがある。したがって、量子化グリッドは多様体の計量（Metric Tensor）に応じて適応的であるべきである。

### **2.2 RSAVQ: Fisher情報量に基づくリーマン量子化**

この問題に対処するアプローチとして、**Riemannian Sensitivity-Aware Vector Quantization (RSAVQ)** 5 が提案されている。RSAVQは、パラメータ空間の局所的な幾何構造を捉えるためにFisher情報行列（FIM）をリーマン計量として利用する。

#### **2.2.1 誤差の方向制御**

RSAVQの重要な洞察は、量子化誤差 $\\epsilon \= w\_{quant} \- w\_{cont}$ をユークリッドノルム $\\|\\epsilon\\|\_2$ で最小化するのではなく、損失関数に対する感度を考慮したリーマンノルム $\\|\\epsilon\\|\_{g} \= \\sqrt{\\epsilon^\\top G(w) \\epsilon}$ で制御することである。  
ここで、$G(w)$ はFIMである。具体的には、量子化誤差ベクトルを「負の自然勾配方向（Natural Gradient Direction）」に沿って射影することで、モデルの出力に対する悪影響を最小化する 5。

* **1.58bitへの適用:** 3値量子化を行う際、単純な丸め処理（Rounding）ではなく、FIMによって定義される楕円体の中で最も近い離散点を選ぶ、あるいは誤差が損失関数の等高線に沿う（損失を変化させない）方向に逃げるように量子化を行うべきである。

### **2.3 量子化リーマン勾配追跡法（Q-RGT）**

量子化された勾配を用いて多様体上で最適化を行うためのアルゴリズムとして、Quantized Riemannian Gradient Tracking (Q-RGT) 6 がある。  
通常、多様体上の最適化では、接空間上のベクトルを多様体上に戻す「レトラクション（Retraction）」操作が必要となる。Q-RGTの研究は、量子化ノイズが存在する場合でも、適切な条件下では $O(1/K)$ の収束率（非量子化手法と同等）を達成できることを示している。

#### **2.3.1 アルゴリズムの要点**

1. **分散最適化の文脈:** Q-RGTは元々分散学習における通信ボトルネック解消のために提案されたが、その数学的構造は「重みの量子化」にも転用可能である。  
2. **量子化雑音の扱い:** 量子化操作を $Q(v) \= v \+ \\xi$ （$\\xi$はノイズ）とモデル化し、このノイズが有界かつ不偏（Unbiased）であれば、勾配追跡項（Gradient Tracker）が真の勾配を近似し続けることができる。  
3. **レトラクションの回避:** 興味深いことに、量子化ノイズの導入によって、厳密なリーマン射影（Retraction）の制約を緩和し、計算効率を向上させる可能性が示唆されている 6。

## ---

**第3章 双曲空間へのMuonオプティマイザの拡張**

大規模言語モデルの学習において、AdamWに代わるオプティマイザとして注目される「Muon（Momentum Orthogonalized optimizer）」は、重み行列の更新を直交化（Orthogonalize）することで、学習の安定性と収束速度を向上させる 7。これを双曲空間（非ユークリッド空間）に拡張するには、「直交性」の定義を再考する必要がある。

### **3.1 J-直交性とLorentz群 $O(n,1)$**

双曲ニューラルネットワーク（HNN）の重み行列は、しばしばLorentz群 $O(n, 1)$ の元として、あるいはLorentz変換として作用する。Lorentzモデルにおける内積は、Minkowski計量 $J \= \\text{diag}(-1, 1, \\dots, 1)$ （または符号を逆にしたもの）によって定義される。  
したがって、ユークリッド空間における直交行列 $Q^\\top Q \= I$ に対応する概念は、双曲空間では J-直交行列（J-orthogonal matrix） である 9。

$$Q^\\top J Q \= J$$

「Hyperbolic Muon」オプティマイザは、更新ステップがこのJ-直交性を（近似的にでも）満たすように設計されなければならない。

### **3.2 一般化極分解（Generalized Polar Decomposition）**

Muonオプティマイザの核心は、勾配（またはモーメンタム）行列 $G$ の極分解 $G \= U P$ （$U$は直交行列、$P$は正定値対称行列）を行い、スケール成分 $P$ を捨てて $U$ のみを更新に用いることにある（Spectral Normalizationの一種）。  
双曲空間においては、これを 一般化極分解（Generalized Polar Decomposition, GPD） に置き換える必要がある 11。  
任意の非特異行列 $A$ は、$A \= W S$ と分解できる。ここで、

* $W$ は **J-直交行列**（$W^\\top J W \= J$）  
* $S$ は J-自己随伴行列（$S^\\top J \= J S$）  
  である。この $W$ を抽出することで、双曲空間の計量を保存する「回転」成分のみを取り出した更新が可能となる。

### **3.3 Higham-Schulz反復法の実装**

ユークリッドMuonでは、SVDを行わずに直交行列を近似するためにNewton-Schulz反復 $X\_{k+1} \= \\frac{1}{2}X\_k(3I \- X\_k^\\top X\_k)$ を用いる 14。  
双曲空間（J-直交行列）のための反復法として、Highamらによって提案された Higham-Schulz反復 が存在する 9。  
数式（J-直交化のためのNewton-Schulz反復）:

$$X\_{k+1} \= \\frac{1}{2} X\_k (3I \- J X\_k^\\top J X\_k)$$

この反復は、行列 $X$ をその一般化極分解のJ-直交因子 $W$ に二次収束させる。  
**Hyperbolic Muonの実装アルゴリズム案:**

1. **勾配取得:** 接空間上の勾配 $G\_t$ を計算する。  
2. **J-直交化:** $G\_t$ に対して Higham-Schulz 反復を数回（通常5回程度）適用し、J-直交行列 $\\tilde{G}\_t$ を得る。  
3. **更新:** 重み $W\_t$ に対し、$\\tilde{G}\_t$ を用いて更新を行う。双曲空間上の「平行移動」として、$W\_{t+1} \= \\text{Exp}\_{W\_t}(-\\eta \\tilde{G}\_t)$、あるいはレトラクション近似を用いる。

**洞察:** この手法により、Muonの持つ「全固有値に対する均一な学習速度」という利点を維持しつつ、双曲空間の幾何学的制約（境界への到達など）を自然に扱うことが可能になる。特に、Higham-Schulz反復は行列積のみで構成されるため、GPU上での並列計算効率が極めて高い 16。

## ---

**第4章 Hyperbolic Normalization：階層性を殺さない正規化**

通常のBatch Normalization（BN）やLayer Normalization（LN）は、データを平均0、分散1に強制する。しかし、双曲空間において「原点からの距離」は「階層の深さ」や「具体性」を意味する重要な情報である。安易なセンタリングは、葉ノード（具体的概念）を根ノード（一般的概念）に引き戻してしまい、モデルの表現力を破壊する 1。

### **4.1 Lorentz Batch Normalization (LBN)**

この問題を解決するために、**Lorentz Batch Normalization (LBN)** 17 が提案されている。LBNは、ユークリッド的な算術平均ではなく、リーマン幾何学的な重心（Fréchet mean）を用いる。

#### **4.1.1 Lorentz Centroid（Fréchet Mean）**

Lorentzモデル $\\mathbb{L}^n$ 上の点集合 $\\{x\_1, \\dots, x\_B\\}$ に対する重心 $\\mu$ は、以下の目的関数を最小化する点として定義される 2。

$$\\mu \= \\arg\\min\_{y \\in \\mathbb{L}^n} \\sum\_{i=1}^B d\_\\mathbb{L}^2(x\_i, y)$$

ここで、$d\_\\mathbb{L}$ はLorentz距離である。Lorentzモデルにおいては、この重心は閉形式（Closed-form）で計算可能であり、反復計算を必要としない点がPoincareモデルに対する大きな利点である。

#### **4.1.2 平行移動によるセンタリング**

LBNの処理フローは以下の通りである 17。

1. **重心計算:** バッチ内のLorentz重心 $\\mu\_B$ を計算する。  
2. **センタリング（Parallel Transport）:** 単純な引き算ではなく、$\\mu\_B$ を双曲空間の原点（North Pole, $\\mathbf{0} \= \[1, 0, \\dots, 0\]^\\top$）に移動させるアイソメトリー（等長変換）$T\_{\\mu\_B \\to \\mathbf{0}}$ をバッチ内の全点に適用する。これにより、相対的な位置関係（幾何構造）を保ったまま分布の中心を原点に合わせる。  
3. **接空間でのスケーリング:** 原点の接空間 $T\_{\\mathbf{0}}\\mathbb{L}^n$ はユークリッド空間と見なせるため、ここで分散の正規化（スケーリング）を行う。  
4. **バイアス移動:** 学習可能なパラメータ（バイアス）である双曲空間上の点 $\\beta$ へ、再び平行移動させる。

PyTorch実装のヒント:  
Lorentzモデルにおける平行移動は、回転行列とブースト行列の組み合わせで表現できる。GeooptやMcTorchライブラリ 18 には、対数写像（Log map）と指数写像（Exp map）が実装されており、これらを組み合わせることで平行移動を実装できる。

$$\\text{Transport}(v) \= \\text{Exp}\_{\\mathbf{0}}(R \\cdot \\text{Log}\_{\\mu\_B}(v))$$

（※厳密にはLorentz変換行列を構成する方が計算効率が良い場合がある 20）。

### **4.2 Poincare Normとの比較**

Poincare Norm 1 は、Poincare球の半径方向のみを調整する簡易的な手法であるが、境界付近での数値安定性に欠ける。対してLBNはLorentzモデル（非有界）で計算するため、1.58bitのような低精度計算においてもオーバーフロー/アンダーフローに対する耐性が高く、勾配消失問題（Vanishing Gradient）を効果的に抑制できる 17。

## ---

**第5章 物理法則の書き換え：Forward/Backwardの再定義**

### **5.1 測地線誤差逆伝播（Geodesic Backpropagation）**

従来の誤差逆伝播（Backprop）は、連鎖律に基づくベクトルの線形結合（引き算）である。しかし、多様体上での誤差 $\\delta$ は接空間のベクトルであり、これを別の層へ伝播させるには、接続（Connection）に従った移動が必要である。  
Geodesic Backpropagation (Geodesic-BP) 21 は、主に心臓電気生理学のEikonal方程式の逆問題において開発された手法だが、これをニューラルネットワークに応用する。

#### **5.1.1 アルゴリズムの概要**

Geodesic-BPでは、学習を「始点（現在の重み）から終点（目標とする重み）への最適な測地線を見つける問題（Geodesic Shooting）」として定式化する 24。

* **Forward:** 入力 $x$ に対し、重み空間上の測地線に沿って変換を行い、出力 $y$ を得る。  
* Backward: 損失 $L$ の勾配 $\\nabla L$ は、ユークリッド空間でのベクトルではなく、リーマン計量 $G^{-1}$ でスケーリングされた共変ベクトル（Covector）として解釈される。

  $$\\text{grad}\_R L \= G^{-1}(W) \\cdot \\nabla\_{Euclid} L$$

  更新ステップは $W\_{new} \= \\text{Exp}\_{W}(-\\eta \\cdot \\text{grad}\_R L)$ となる。

McTorch/Geooptによる近似:  
厳密な平行移動は計算コストが高いため、実際には「レトラクション（Retraction）」を用いた近似が一般的である。レトラクションは、接ベクトルを多様体上に引き戻す一次近似写像であり、計算効率が良い。

* **実装:** PyTorchのautogradフックを利用し、各層の勾配計算直後にRiemannianGradient補正（計量の逆行列を掛ける、あるいは接空間への射影を行う）を挿入する 18。

### **5.2 Stochastic Resonance（確率共鳴）と1.58bit勾配**

1.58bit（3値）ネットワークの最大の課題は、量子化による勾配の消失（Vanishing Gradient due to discretization）である。勾配が量子化閾値（例: 0.5）よりも小さい場合、重みは更新されず、学習が停滞する。  
ここで、確率共鳴（Stochastic Resonance, SR） の理論を応用する 26。SRとは、非線形システムに適切なノイズを加えることで、閾値以下の微弱な信号の検出能力が向上する現象である。

#### **5.2.1 Stochastic RoundingとSRの等価性**

BitNet等の最近の研究で採用されている Stochastic Rounding（確率的丸め） 3 は、まさにこのSRの実装そのものであると解釈できる。  
決定論的丸め（Round）ではなく、以下の確率で丸める：

$$P(x\_q \= \\lfloor x \\rfloor \+ 1\) \= x \- \\lfloor x \\rfloor$$

これにより、期待値の意味で $\\mathbb{E}\[x\_q\] \= x$ が保たれる。

* **メカニズム:** 微小な勾配 update $\\Delta w$ が発生した際、決定論的丸めでは重みは変化しない（$0 \\to 0$）。しかし、Stochastic Roundingでは、$\\Delta w$ に比例して「ビットが反転する確率」が変化する。多数のイテレーション（またはバッチサイズ）を通じて、この確率的な「ゆらぎ」が積分され、正しい勾配方向への移動（トンネル効果）が実現される。

#### **5.2.2 ノイズ分布の最適化**

文献 27 によれば、信号（勾配）の分布がガウス分布でない場合（Transformersの勾配はHeavy-tailedであることが多い）、ガウスノイズよりも Levy flightノイズ（コーシー分布など） のような裾の重いノイズを加えることで、共鳴効果（勾配の復元効率）が最大化される可能性がある。  
したがって、1.58bit化のForwardパスにおいて、単純な一様乱数ではなく、勾配分布の統計量に基づいたノイズを注入する「適応的SR層」の実装が、超低精度学習の収束を加速させる鍵となる。

## ---

**第6章 Koopman作用素による力学系の一貫性と安定化**

リカレントな構造（RNN, SSM, Transformerの自己回帰）を持つモデルを双曲空間で学習させる際、非線形性が強いため、長時間推論において状態が発散したり、カオス的挙動を示したりするリスクがある。**Koopman作用素理論**は、非線形力学系を無限次元の線形作用素として扱うことで、大域的な安定解析を可能にする 29。

### **6.1 スペクトル安定性の保証**

Koopman作用素 $K$ の有限次元近似行列を学習する際、その固有値 $\\lambda\_i$ の分布を制御することでシステムの安定性を保証できる。  
推論が発散しないための条件は、すべての固有値が単位円内にあること（$|\\lambda\_i| \\le 1$）である。

#### **6.1.1 Eigenloss（固有値正則化）**

学習中の損失関数に、Koopman行列のスペクトル半径を制限する項を追加する Eigenloss 31 を導入する。

$$\\mathcal{L}\_{stable} \= \\sum\_{i=1}^m \\max(0, |\\lambda\_i(K)| \- 1)$$

また、固有値の虚部をペナルティ化することで、高周波振動を抑制することも可能である。

* **Lyapunov制約:** さらに強力な安定性を得るために、ニューラルネットワークでパラメータ化されたLyapunov関数 $V(x)$ を同時に学習し、$V(Kx) \< V(x)$ という制約を課す手法もある 33。これにより、モデルの「安定領域（Region of Attraction）」を明示的に最大化できる。

### **6.2 整合性正則化（Consistency Regularization）**

Koopman埋め込み空間での線形推移 $z\_{t+1} \= K z\_t$ と、元の非線形空間での推移 $x\_{t+1} \= f(x\_t)$ の整合性を保つため、Consistency Loss を導入する。

$$\\mathcal{L}\_{consist} \= \\| \\phi(x\_{t+1}) \- K \\phi(x\_t) \\|^2$$

ここで $\\phi$ は観測関数（エンコーダ）である。この項は、モデルが物理的な（あるいは論理的な）一貫性を保ちながら長期記憶を保持するために不可欠である 34。

## ---

**第7章 メタ学習と自己組織化：パラメータ自体の動的制御**

最後に、これらの複雑な要素（曲率、学習率、SRノイズレベル）を人手で調整することは不可能に近い。**Meta-Learning** を用いて、ハイパーパラメータ自体を「学習可能なテンソル」として扱う。

### **7.1 Meta-SGD: テンソル学習率**

Meta-SGD 35 は、重みの初期値だけでなく、パラメータごとの学習率 $\\alpha$ も学習対象とする。

$$\\theta\_{new} \= \\theta \- \\alpha \\odot \\nabla L$$

ここで $\\odot$ は要素ごとの積（Hadamard product）である。

* **幾何学的解釈:** リーマン最適化において、学習率テンソル $\\alpha$ は「局所的な計量のスケーリングファクター（Preconditioner）」として機能する。Meta-SGDを用いることで、多様体の曲率が激しい場所では小さなステップを、平坦な場所では大きなステップを自動的に選択するよう自己組織化される。

### **7.2 Self-Tuning Networks (STN) と動的曲率**

**Self-Tuning Networks (STN)** 36 は、現在の学習状態（損失、勾配ノルムなど）を入力とし、ハイパーパラメータを出力するHypernetworkを組み込む手法である。

* **動的曲率学習（Dynamic Curvature Learning）:** 双曲空間の曲率 $c$ （負の値）を固定せず、STNによって動的に変化させる 38。学習初期は $c \\approx 0$（ユークリッド空間に近い状態）で大域的な探索を行い、学習が進むにつれて $|c|$ を大きくして階層構造を「尖らせる」というカリキュラム学習的なアプローチが可能になる。これにより、最適化の難易度を緩和しつつ、最終的な表現力を最大化できる。

## ---

**第8章 結論：推奨されるアーキテクチャの仕様**

以上の調査に基づき、以下の仕様を持つ「物理幾何学的1.58bitモデル」の開発を推奨する。

1. **多様体基盤**: Lorentzモデル $\\mathbb{L}^n$（数値安定性のためPoincare球より推奨）。  
2. **重み表現**: 一般化極分解形式 $W \= U\_{J} S$ で保持し、更新は $U\_{J}$（Lorentz回転成分）に対して行う。  
3. **オプティマイザ**: **Hyperbolic Muon**。勾配に対しHigham-Schulz反復（J-直交化）を適用し、更新ベクトルを生成。  
4. **量子化**: **RSAVQ**に基づく不均一コードブック、または**Stochastic Resonance**（確率的丸め \+ Levyノイズ）を用いた1.58bit化。  
5. **正規化**: **Lorentz Batch Norm**。重心の平行移動と接空間スケーリングを実装。  
6. **安定化**: **Koopman Eigenloss**により、リカレント状態のスペクトル半径を1以下に抑制。  
7. **メタ制御**: **Meta-SGD**で学習率をテンソル化し、**STN**で曲率を動的にアニーリング。

このアーキテクチャは、計算効率（1.58bit）、表現力（双曲幾何）、安定性（Koopman/Lyapunov）を数学的に矛盾なく統合した、次世代の基盤モデルとなり得る。

#### **引用文献**

1. Hierarchical Mamba Meets Hyperbolic Geometry: A New Paradigm for Structured Language Embeddings \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2505.18973v3](https://arxiv.org/html/2505.18973v3)  
2. Lorentzian Distance Learning for Hyperbolic Representations, 12月 8, 2025にアクセス、 [https://proceedings.mlr.press/v97/law19a/law19a.pdf](https://proceedings.mlr.press/v97/law19a/law19a.pdf)  
3. Direct Quantized Training of Language Models with Stochastic Rounding \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2412.04787v3](https://arxiv.org/html/2412.04787v3)  
4. Asymptotic quantization on Riemannian manifolds via covering growth estimates \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/abs/2402.13164](https://arxiv.org/abs/2402.13164)  
5. RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2510.01240v1](https://arxiv.org/html/2510.01240v1)  
6. Decentralized Optimization on Compact Submanifolds by Quantized Riemannian Gradient Tracking | Request PDF \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/391582780\_Decentralized\_Optimization\_on\_Compact\_Submanifolds\_by\_Quantized\_Riemannian\_Gradient\_Tracking](https://www.researchgate.net/publication/391582780_Decentralized_Optimization_on_Compact_Submanifolds_by_Quantized_Riemannian_Gradient_Tracking)  
7. An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2510.09827v1](https://arxiv.org/html/2510.09827v1)  
8. Daily Papers \- Hugging Face, 12月 8, 2025にアクセス、 [https://huggingface.co/papers?q=Muon%20optimizer](https://huggingface.co/papers?q=Muon+optimizer)  
9. J-Orthogonal Matrices: Properties and Generation | SIAM Review, 12月 8, 2025にアクセス、 [https://epubs.siam.org/doi/10.1137/S0036144502414930](https://epubs.siam.org/doi/10.1137/S0036144502414930)  
10. J-Orthogonal Matrices: Properties and Generation \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/30045389\_J-Orthogonal\_Matrices\_Properties\_and\_Generation](https://www.researchgate.net/publication/30045389_J-Orthogonal_Matrices_Properties_and_Generation)  
11. The Canonical Generalized Polar Decomposition \- MIMS EPrints \- The University of Manchester, 12月 8, 2025にアクセス、 [https://eprints.maths.manchester.ac.uk/1490/01/covered/MIMS\_ep2009\_52.pdf](https://eprints.maths.manchester.ac.uk/1490/01/covered/MIMS_ep2009_52.pdf)  
12. Polar Decompositions of Normal Operators in Indefinite Inner Product Spaces, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/226699900\_Polar\_Decompositions\_of\_Normal\_Operators\_in\_Indefinite\_Inner\_Product\_Spaces](https://www.researchgate.net/publication/226699900_Polar_Decompositions_of_Normal_Operators_in_Indefinite_Inner_Product_Spaces)  
13. Stable and Efficient Computation of Generalized Polar Decompositions \- SIAM.org, 12月 8, 2025にアクセス、 [https://epubs.siam.org/doi/10.1137/21M1411986](https://epubs.siam.org/doi/10.1137/21M1411986)  
14. Newton-Schulz \- docs.modula.systems, 12月 8, 2025にアクセス、 [https://docs.modula.systems/algorithms/newton-schulz/](https://docs.modula.systems/algorithms/newton-schulz/)  
15. Algorithms for the Polar Decomposition \- Sci-Hub, 12月 8, 2025にアクセス、 [https://2024.sci-hub.se/1856/7952865f130fa1dde12105f0dcbf2980/gander1990.pdf](https://2024.sci-hub.se/1856/7952865f130fa1dde12105f0dcbf2980/gander1990.pdf)  
16. Task-Based Polar Decomposition Using SLATE on Massively Parallel Systems with Hardware Accelerators \- The Netlib, 12月 8, 2025にアクセス、 [https://www.netlib.org/utk/people/JackDongarra/PAPERS/task-polor-sc23.pdf](https://www.netlib.org/utk/people/JackDongarra/PAPERS/task-polor-sc23.pdf)  
17. FULLY HYPERBOLIC CONVOLUTIONAL NEURAL ... \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/pdf?id=ekz1hN5QNh](https://openreview.net/pdf?id=ekz1hN5QNh)  
18. (PDF) McTorch, a manifold optimization library for deep learning \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/328063619\_McTorch\_a\_manifold\_optimization\_library\_for\_deep\_learning](https://www.researchgate.net/publication/328063619_McTorch_a_manifold_optimization_library_for_deep_learning)  
19. TpG Geoopt: Riemannian Optimization in PyTorch \- Graph Representation Learning and Beyond (GRL+), 12月 8, 2025にアクセス、 [https://grlplus.github.io/papers/93.pdf](https://grlplus.github.io/papers/93.pdf)  
20. Optimal alignment of Lorentz orientation and generalization to matrix Lie groups \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2506.14994v3](https://arxiv.org/html/2506.14994v3)  
21. Digital twinning of cardiac electrophysiology models from the surface ECG: a geodesic backpropagation approach \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/pdf/2308.08410](https://arxiv.org/pdf/2308.08410)  
22. (PDF) Digital twinning of cardiac electrophysiology models from the surface ECG: a geodesic backpropagation approach \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/373164106\_Digital\_twinning\_of\_cardiac\_electrophysiology\_models\_from\_the\_surface\_ECG\_a\_geodesic\_backpropagation\_approach](https://www.researchgate.net/publication/373164106_Digital_twinning_of_cardiac_electrophysiology_models_from_the_surface_ECG_a_geodesic_backpropagation_approach)  
23. Digital Twinning of Cardiac Electrophysiology Models From the Surface ECG: A Geodesic Backpropagation Approach \- iris@unitn, 12月 8, 2025にアクセス、 [https://iris.unitn.it/retrieve/750aeae9-236d-4010-a206-6990e6038594/2024%20Grandits%20-%20GeodesicBP%20%28TMBE%29.pdf](https://iris.unitn.it/retrieve/750aeae9-236d-4010-a206-6990e6038594/2024%20Grandits%20-%20GeodesicBP%20%28TMBE%29.pdf)  
24. Kinetic Energy Fields: A Solution of Riemannian ... \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/pdf?id=5P1cKfotkC](https://openreview.net/pdf?id=5P1cKfotkC)  
25. The Riemannian Geometry of Deep Generative Models \- CVF Open Access, 12月 8, 2025にアクセス、 [https://openaccess.thecvf.com/content\_cvpr\_2018\_workshops/papers/w10/Shao\_The\_Riemannian\_Geometry\_CVPR\_2018\_paper.pdf](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Shao_The_Riemannian_Geometry_CVPR_2018_paper.pdf)  
26. Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2510.03224v1](https://arxiv.org/html/2510.03224v1)  
27. Stochastic Resonance Can Enhance Information Transmission in Neural Networks | Request PDF \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/50850400\_Stochastic\_Resonance\_Can\_Enhance\_Information\_Transmission\_in\_Neural\_Networks](https://www.researchgate.net/publication/50850400_Stochastic_Resonance_Can_Enhance_Information_Transmission_in_Neural_Networks)  
28. Direct Quantized Training of Language Models with Stochastic Rounding \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2412.04787v2](https://arxiv.org/html/2412.04787v2)  
29. Learning Koopman-based Stability Certificates for Unknown Nonlinear Systems \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2412.02807v2](https://arxiv.org/html/2412.02807v2)  
30. Notes on Koopman Operator Theory \- UK Fluids Network, 12月 8, 2025にアクセス、 [https://fluids.ac.uk/files/meetings/KoopmanNotes.1575558616.pdf](https://fluids.ac.uk/files/meetings/KoopmanNotes.1575558616.pdf)  
31. Eigenvalue Initialisation and Regularisation for Koopman ... \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/pdf/2212.12086](https://arxiv.org/pdf/2212.12086)  
32. EIGENVALUE INITIALISATION AND REGULARISATION FOR KOOPMAN AUTOENCODERS \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/references/pdf?id=SXPRcxRCw](https://openreview.net/references/pdf?id=SXPRcxRCw)  
33. Physics-Informed Machine Learning for Characterizing System Stability \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/html/2511.08831v1](https://arxiv.org/html/2511.08831v1)  
34. FlowMixer: A Depth-Agnostic Neural Architecture for Interpretable Spatiotemporal Forecasting \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/pdf/08a3fcab53889e2469399a667a998a7d271939b9.pdf](https://openreview.net/pdf/08a3fcab53889e2469399a667a998a7d271939b9.pdf)  
35. Meta-SGD: Learning to Learn Quickly for Few Shot Learning \- ResearchGate, 12月 8, 2025にアクセス、 [https://www.researchgate.net/publication/318813512\_Meta-SGD\_Learning\_to\_Learn\_Quickly\_for\_Few\_Shot\_Learning](https://www.researchgate.net/publication/318813512_Meta-SGD_Learning_to_Learn_Quickly_for_Few_Shot_Learning)  
36. SELF-TUNING NETWORKS: BILEVEL OPTIMIZATION OF HYPERPARAMETERS US \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/pdf?id=r1eEG20qKQ](https://openreview.net/pdf?id=r1eEG20qKQ)  
37. self-tuning networks: bilevel optimization of hyperparameters us \- arXiv, 12月 8, 2025にアクセス、 [https://arxiv.org/pdf/1903.03088](https://arxiv.org/pdf/1903.03088)  
38. Stepping on the Edge: Curvature Aware Learning Rate Tuners \- OpenReview, 12月 8, 2025にアクセス、 [https://openreview.net/attachment?id=SEflLHIhhJ\&name=pdf](https://openreview.net/attachment?id=SEflLHIhhJ&name=pdf)