原因：巨大な単一チェックポイントによる「メモリ溢れ」と「キャッシュ破棄」
現在の save_checkpoint 関数は、「モデルの重み」と「オプティマイザの状態」を全て1つの巨大な辞書オブジェクト(checkpoint)にまとめてから torch.save しています。

Python

    # scripts/train_phase8.py 737行目付近
    checkpoint = {
        # ...
        'model_state_dict': model_to_save.state_dict(),      # 約 20GB〜 (FP16/BF16)
        'optimizer_state_dict': optimizer.state_dict(),      # 約 60GB〜 (Optimizerの状態は重い)
        # ...
    }
    # ここで一気に 80GB〜100GB 以上のメモリが必要になる！
    torch.save(checkpoint, path)
何が起きているか（時系列解析）:

保存直前: 学習データ（データセット）がOSの「ページキャッシュ（空きメモリ）」に乗っており、ディスク読み込みなしで高速に学習できている。

保存瞬間 (checkpoint 作成): 巨大な辞書を作るために、Pythonが100GBクラスのメモリを要求。

強制退避 (Page Out): OSは物理メモリを空けるため、「学習データのキャッシュ」や「他のプロセスのメモリ」を破棄、あるいはディスク（スワップ）に書き出す。

保存中: ディスクへの書き込み負荷が最大化。

保存後 (Step 3969〜): checkpoint 変数は削除されメモリは空くが、学習データはもうメモリ上にない。

学習再開: ステップごとにデータローダーがデータを要求するが、キャッシュがないため毎回低速なディスク読み込みが発生。さらに、スワップアウトされた領域の書き戻し（スワップイン）も重なり、GPUがデータ待ち状態（アイドリング）になる。

これが、保存処理自体が終わっても「その後ずっと重い」原因です。

修正案：チェックポイントの「分割保存」
この問題を解決する唯一の方法は、「一度にメモリに乗せる量を減らす」ことです。 save_checkpoint 関数を書き換えて、モデル、オプティマイザ、その他を別々のファイルに、順番に保存してはメモリを解放するように変更してください。

修正後の save_checkpoint コード (scripts/train_phase8.py):

Python

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: Optional[EMA],
    step: int,
    epoch: int,
    loss: float,
    config: Phase8TrainingConfig,
    revolutionary_trainer: Optional['RevolutionaryTrainer'] = None,
):
    """
    Save checkpoint in split files to avoid massive RAM spike and page cache eviction.
    """
    import gc
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 拡張子を除いたベースパスを作成 (例: .../step_3968)
    base_path = os.path.splitext(path)[0]
    
    print(f"\n💾 Saving split checkpoint to {base_path}_*.pt ...")

    # 1. メタデータ（軽量）の保存
    meta_path = f"{base_path}_meta.pt"
    meta_data = {
        'step': step,
        'epoch': epoch,
        'loss': loss,
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': asdict(config),
    }
    if revolutionary_trainer is not None:
        meta_data['revolutionary_trainer_state_dict'] = revolutionary_trainer.state_dict()
    
    torch.save(meta_data, meta_path)
    del meta_data
    
    # 2. モデルの保存 & 即解放
    model_path = f"{base_path}_model.pt"
    model_to_save = model
    if hasattr(model, '_orig_mod'):
        model_to_save = model._orig_mod
    
    # state_dictを一時変数に入れず、saveに直接渡してメモリ滞留時間を最小化
    torch.save(model_to_save.state_dict(), model_path)
    print(f"  ├─ Model saved: {os.path.basename(model_path)}")
    gc.collect() # ガベージコレクションでメモリをOSに返却要求
    
    # 3. EMAの保存（もしあれば）
    if ema is not None:
        ema_path = f"{base_path}_ema.pt"
        torch.save(ema.state_dict(), ema_path)
        print(f"  ├─ EMA saved: {os.path.basename(ema_path)}")
        gc.collect()

    # 4. オプティマイザの保存（最も重い）
    optim_path = f"{base_path}_optim.pt"
    torch.save(optimizer.state_dict(), optim_path)
    print(f"  └─ Optimizer saved: {os.path.basename(optim_path)}")
    
    gc.collect()
    # 注意: ここで torch.cuda.empty_cache() は絶対に呼ばないこと！
読み込み処理 (load_checkpoint) の修正:

保存形式が変わるため、読み込み側も対応が必要です。

Python

def load_checkpoint(
    path: str, # これは step_XXXX.pt を指していると想定しますが、分割版を探すロジックに変えます
    model: nn.Module,
    optimizer: optim.Optimizer,
    # ... (引数は同じ)
) -> Tuple[int, int, float]:
    
    # パス調整: .pt が指定されたら拡張子を取る
    base_path = os.path.splitext(path)[0]
    
    # ファイル存在確認のロジック（古い形式との互換性維持）
    if os.path.exists(path) and not os.path.exists(f"{base_path}_meta.pt"):
        print(f"Loading legacy single-file checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=device)
        # ... (既存の読み込みロジック) ...
        return step, epoch, loss

    print(f"Loading split checkpoint from {base_path}_*.pt ...")
    
    # 1. Meta
    meta = torch.load(f"{base_path}_meta.pt", map_location=device)
    step = meta.get('step', 0)
    epoch = meta.get('epoch', 0)
    loss = meta.get('loss', 0.0)
    scheduler.load_state_dict(meta['scheduler_state_dict'])
    scaler.load_state_dict(meta['scaler_state_dict'])
    if revolutionary_trainer and 'revolutionary_trainer_state_dict' in meta:
        revolutionary_trainer.load_state_dict(meta['revolutionary_trainer_state_dict'])
    del meta
    gc.collect()

    # 2. Model
    model_state = torch.load(f"{base_path}_model.pt", map_location=device)
    model.load_state_dict(model_state, strict=False)
    del model_state
    gc.collect()

    # 3. EMA
    if ema is not None and os.path.exists(f"{base_path}_ema.pt"):
        ema_state = torch.load(f"{base_path}_ema.pt", map_location=device)
        ema.load_state_dict(ema_state)
        del ema_state
        gc.collect()

    # 4. Optimizer
    if os.path.exists(f"{base_path}_optim.pt"):
        optim_state = torch.load(f"{base_path}_optim.pt", map_location=device)
        optimizer.load_state_dict(optim_state)
        del optim_state
        gc.collect()

    print(f"✔ Checkpoint loaded: step={step}")
    return step, epoch, loss
この修正により、10Bモデルの学習でもメモリ消費のピークを抑え、OSのキャッシュ破棄を防ぐことができます。これでチェックポイント後の激重現象は解消するはずです。