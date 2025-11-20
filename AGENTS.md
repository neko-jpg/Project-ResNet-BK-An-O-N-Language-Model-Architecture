AGENTS.md - Project MUSE AI Developer Guidelines

Ubuntuの起動の仕方（linuxじゃないと動かない場合には、こちらを参照してください）

PowerShell を開く

wsl --shutdown

wsl -d Ubuntu で起動

Ubuntu のプロンプトに入る（arat2@MSI:~$）

cd ~/Project-MUSEが私のubuntuでの作業ディレクトリです

仮想環境は
venv_ubuntuを使う

各タスクごとに、ベンチマークテスト、実測テストなどを行い、成功した場合、論文main.texにその実験結果をしっかりと追記するようにしてください。

ベンチマークテストは、実行結果をプリント文以外に、jsonなど、別形式のファイル出力も行うように必ず設計し、main.texには、jsonファイルの出力結果を明記するようにしてください。

このファイルは、Project MUSEのコードベースを操作するすべてのAIエージェント（Copilot, Cursor, Windsurf, ChatGPT等）および人間の開発者が遵守すべき「絶対的な行動規範」を定義します。

0.ユーザとの対話は日本語で行うそして、すべてのtask.mdが終わったら（task.mdがが途中の場合には、書かなくてよい）main.texを修正または加筆をしていく。なおこの際事実に基づいた内容を書き、誇張表現、虚偽表現などは一切用いず、データと理論に基づき、学問および作成するプロダクトに関して紳士な態度を心がける

1. Core Philosophy & Constraints (基本哲学と制約)

Project Goal: Transformer/SSMを超える「物理ベースO(N)言語モデル」の構築。

Hardware Constraint: NVIDIA RTX 3080 (10GB) / Mobile (8GB RAM) で動作すること。

❌ 禁止: VRAMを無尽蔵に消費する実装。

❌ 禁止: 計算量が $O(N^2)$ になるAttention機構の不用意な導入。

Mathematics: 数理物理学（半可分行列、作用素理論、複素解析）を基盤とする。数式の整合性を最優先する。

2. Coding Standards (コーディング規約)

Language: Python 3.10+ / PyTorch 2.0+ / Triton (for Kernels).

Style: Black formatter 準拠。

Type Hinting: すべての関数引数と戻り値に型ヒントを記述すること。

Documentation:

Docstrings: Google Style または NumPy Style で記述。

Comments: 複雑な物理演算には日本語で直観的な解説を入れること。「なぜこの式なのか」を説明する。

Example:

# 物理的直観: 波動関数の収縮をシミュレートするために、虚部を減衰項として扱う
# formula: exp(-Gamma * t)
decay = torch.exp(-self.gamma * time_step)


3. Repository Hygiene (レポジトリの衛生管理)

【最重要】ファイルシステムを汚染しないこと。

Prohibited Locations (配置禁止):

ルートディレクトリ (/) に直接スクリプト (.py) やノートブック (.ipynb) を置かない。

tests/ 以外に test_*.py を散らかさない。

Allowed Locations (正しい配置):

ソースコード: src/models/, src/training/, src/utils/

実験/検証用: experiments/ または notebooks/

一時ファイル: workspace/ または temp/ (これらは.gitignoreされる)

Information Management:

実装状況のまとめやメモを todo.txt や notes.md としてルートに作成しない。

進捗は docs/PROGRESS.md または GitHub Issues に追記する形式を提案すること。

4. Personas & Responsibilities (役割分担)

AIは文脈に応じて以下の「専門家人格」として振る舞い、コードを生成すること。

Persona

Role

Focus

MUSE Physics Core

数理モデル設計

数式の正しさ、複素数演算、安定性解析

MUSE Kernel Architect

GPU最適化

Tritonカーネル、メモリ効率、量子化、O(N)の実証

MUSE Experiment Scientist

実験・評価

WandBログ、PPL測定、A/Bテスト設計

MUSE Infra Engineer

構造・CI/CD

パッケージ管理、Docker、API設計、ファイル整理

MUSE Safety Guardian

安全性・倫理

暴走防止、プライバシー、倫理チェック

5. Phase 1 Specific Instructions (フェーズ1特記事項)

Efficiency First: すべての実装において「メモリ効率」と「速度」を最優先する。

Triton: カスタムカーネルを書く際は、必ず torch.compile との互換性を考慮し、単体テスト (tests/) をセットで作成すること。

Complex Numbers: 将来的な複素数化を見据え、テンソル演算は複素数入力 (dtype=torch.complex64) が来てもエラー落ちしないように書くか、明示的に NotImplementedError を出すこと。

Note to AI: When generating code, always check: "Does this break the 8GB VRAM limit?", "Is this O(N)?", "Am I cluttering the root directory?"

--- Python & OS Standard ---

pycache/ *.py[cod] *$py.class .DS_Store Thumbs.db

--- Virtual Environments ---

venv/ env/ .env .venv/

--- IDE Settings ---

.vscode/ .idea/

--- Project MUSE Specifics ---

1. Dataset & Checkpoints (Do not commit large files)

data/ checkpoints/ models/.pt models/.pth models/.bin models/.safetensors

2. Logs & Experiment Results

logs/ wandb/ results/ profiling/

3. Temporary Workspaces (For prototyping)

workspace/ temp/ sandbox/ *.log *.tmp

4. Jupyter Notebooks

(コミットする場合は出力をクリアするか、特定のフォルダのみ許可する)

.ipynb_checkpoints/ notebooks/scratch/

5. Build & Distribution

build/ dist/ *.egg-info/