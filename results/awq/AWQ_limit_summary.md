# AWQ (Qwen3-8B-AWQ / vLLM) 8GB VRAM 限界実測まとめ

実機: RTX 4060 Laptop 8GB (8,188 MiB) / vLLM 0.22.1 / enforce_eager=True / dtype fp16。
モデル: `Qwen/Qwen3-8B-AWQ`（AWQ INT4, ローカル `models/awq/Qwen3-8B-AWQ`）。
ログ原本: 本ディレクトリ `attempt_*.log` / `success_*.log` / `official_*.log`。生CSV: `awq_probe.csv`。

## 結論
**AWQ INT4 (Qwen3-8B) は 8GB VRAM では実験の最小条件 8K すら起動できない。**
4bit重みでも GPU 上で **5.71 GiB** を占有し、vLLM のアクティベーション等オーバーヘッドを差し引くと
KVキャッシュに回せるのは最大でも **0.73 GiB（util=0.97 時）**。これは約 **5,280 トークン**分しかなく、
8,192 トークン（必要KV 1.12 GiB）に届かない。util を上げても 1.0 が上限で、外挿しても 8K には足りない。

## gpu_memory_utilization スイープ（max_model_len=8192 固定）
| util | 確保KV | GPU KVトークン数 | 推定最大コンテキスト長 | 8192起動 | peak VRAM |
|------|--------|------------------|------------------------|----------|-----------|
| 0.90 | 0.19 GiB | ~1,392 | 1,392 | ✗ ValueError | 7.98 GiB |
| 0.93 | 0.42 GiB | ~3,056 | 3,056 | ✗ ValueError | 7.98 GiB |
| 0.95 | 0.57 GiB | ~4,160 | 4,160 | ✗ ValueError | 7.98 GiB |
| 0.97 | 0.73 GiB | 5,280 | 5,280 | ✗ ValueError | 7.98 GiB |

- 重みロードは一貫して 5.71 GiB。peak VRAM は util 設定どおり常に 7.98 GiB（≒8GB上限）に張り付く。
- 必要KVはコンテキスト長に比例: 8K=1.12 / 16K=2.25 / 32K=4.5 GiB。確保0.73 GiB では全条件不足。

## 起動成功した縮小条件（参考: AWQ自体は正常動作することの確認）
| ctx | util | 結果 | gen tok/s | peak VRAM | 出力 |
|-----|------|------|-----------|-----------|------|
| 4096 | 0.97 | **起動成功・推論完了** | 22.67 | 7.98 GiB | 有効JSON |

- KV 5,280トークン枠に 4,096 が収まり、T1(2,591tok)入力で正常にJSON抽出。
- 速度 22.67 tps は同条件帯の GGUF(Qwen3-8B 8K=28.6tps) より遅め。ただし文脈長が違うため厳密比較は不可。

## 実験条件での公式記録（results/results.csv に ERROR 行として収録）
| ctx | util | 必要KV | 確保KV | 結果 |
|-----|------|--------|--------|------|
| 8192  | 0.97 | 1.12 GiB | 0.73 GiB | OOM相当 ValueError（起動失敗） |
| 16384 | 0.97 | 2.25 GiB | 0.73 GiB | OOM相当 ValueError（起動失敗） |
| 32768 | 0.97 | 4.50 GiB | 0.73 GiB | OOM相当 ValueError（起動失敗） |

## 環境上の注意（再現のため）
- vLLM の EngineCore はサブプロセス起動。親プロセスで torch.cuda を触ると
  「Cannot re-initialize CUDA in forked subprocess」。→ benchmark.py は AWQ 時に
  `clear_gpu_cache()` を呼ばず、`VLLM_WORKER_MULTIPROC_METHOD=spawn` を設定。
- FlashInfer サンプラーは ninja/nvcc を要する JIT。本機は未整備のため
  `VLLM_USE_FLASHINFER_SAMPLER=0`（ネイティブサンプラー）で回避。
- 上記2点は benchmark.py の run_awq 内で自動設定済み。
